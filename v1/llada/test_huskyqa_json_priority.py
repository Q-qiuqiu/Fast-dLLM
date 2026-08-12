"""Run one real HuskyQA query against the JSON Agent-priority server.

The script validates that the response can be split into planning reasoning and
normal plan JSON, then records Agent recognition times, token counts, throughput,
and end-to-end latency.
"""

import argparse
import json
import re
from pathlib import Path

import requests


HUSKY_AGENTS = {
    "code_agent",
    "math_agent",
    "search_agent",
    "commonsense_agent",
}

HUSKY_PLANNER_PROMPT = """You are a planning agent. Decompose the user query into
executable subtasks and choose one suitable agent for each subtask.

Available agents:
- code_agent: writes and runs Python code for precise computations.
- math_agent: solves math questions by step-by-step reasoning.
- search_agent: searches the web for external or current facts.
- commonsense_agent: answers with reasoning and general knowledge.

Return both PLAN_JSON and PLANNING_REASONING sections. PLAN_JSON must contain a
valid JSON array. Every plan object must contain agent, id, task, reason, and dep.
Use only the four exact Agent names above. Preserve all important entities,
numbers, constraints, and dates; use dependencies when a later step needs an
earlier result.
"""


def extract_plan(content):
    marker = re.search(r"(?m)^\s*PLAN_JSON\s*:?\s*$", content)
    if marker:
        segment = content[marker.end():]
        end = re.search(r"(?m)^\s*END_PLAN_JSON\s*$", segment)
        if end:
            segment = segment[:end.start()]
    else:
        segment = content
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", segment):
        try:
            value, _ = decoder.raw_decode(segment[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    raise ValueError("No valid plan JSON array was found in the response.")


def extract_reasoning(content):
    start = re.search(r"(?m)^\s*PLANNING_REASONING\s*:?\s*$", content)
    if not start:
        return None
    segment = content[start.end():]
    end = re.search(r"(?m)^\s*END_PLANNING_REASONING\s*$", segment)
    return (segment[:end.start()] if end else segment).strip() or None


def validate_sections(content, reasoning):
    """Require an unambiguous reasoning+plan envelope.

    END_PLANNING_REASONING is optional because end-of-response is an equally
    unambiguous delimiter and LLaDA sometimes emits EOS immediately after the
    reasoning text.  Avoid appending synthetic response tokens solely for a
    redundant closing marker.
    """

    errors = []
    required_markers = (
        "PLAN_JSON",
        "END_PLAN_JSON",
        "PLANNING_REASONING",
    )
    for marker in required_markers:
        matches = re.findall(rf"(?m)^\s*{re.escape(marker)}\s*:?\s*$", content)
        if len(matches) != 1:
            errors.append(f"expected exactly one {marker} marker, found {len(matches)}")
    end_matches = re.findall(
        r"(?m)^\s*END_PLANNING_REASONING\s*:?\s*$", content
    )
    if len(end_matches) > 1:
        errors.append(
            "expected at most one END_PLANNING_REASONING marker, "
            f"found {len(end_matches)}"
        )
    if reasoning is None:
        errors.append("PLANNING_REASONING section is missing or empty")
    return errors


def validate_plan(plan):
    errors = []
    ids = set()
    for index, step in enumerate(plan, start=1):
        missing = [key for key in ("agent", "id", "task", "reason", "dep") if key not in step]
        if missing:
            errors.append(f"step {index}: missing fields {missing}")
        agent = str(step.get("agent", "")).strip()
        if agent not in HUSKY_AGENTS:
            errors.append(f"step {index}: unsupported agent {agent!r}")
        step_id = step.get("id")
        if step_id in ids:
            errors.append(f"step {index}: duplicate id {step_id!r}")
        ids.add(step_id)
        if not isinstance(step.get("dep", []), list):
            errors.append(f"step {index}: dep is not a list")
    return errors


def load_query(path, index):
    with Path(path).open("r", encoding="utf-8") as file:
        rows = json.load(file)
    row = rows[index]
    return row.get("question") or row.get("query"), row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7005/v1")
    parser.add_argument("--api-key", default="empty")
    parser.add_argument("--model", default="/data/labshare/Param/llada")
    parser.add_argument(
        "--input",
        default=(
            "/data/home/yzx/Agent-Oriented-Planning/yzx_test/benchmarks/"
            "huskyqa/huskyqa_raw.json"
        ),
    )
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--output",
        default="huskyqa_json_priority_result.json",
    )
    args = parser.parse_args()

    query, source = load_query(args.input, args.query_index)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": HUSKY_PLANNER_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=args.timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"Planner request failed with HTTP {response.status_code}: {response.text}"
        )

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    plan = None
    split_error = None
    validation_errors = []
    try:
        plan = extract_plan(content)
        validation_errors = validate_plan(plan)
    except Exception as error:  # keep raw output and timings for diagnosis
        split_error = str(error)
    reasoning = extract_reasoning(content)
    format_errors = validate_sections(content, reasoning)
    metrics = data.get("fastdllm") or {}
    usage = data.get("usage") or {}
    agent_metrics = (metrics.get("agent_priority") or {}).get("agent_slots") or []
    expected_priority_agents = [
        str(step.get("agent", "")).strip()
        for step in (plan or [])[:4]
    ]
    recognized_priority_agents = [
        item.get("agent") for item in agent_metrics[:len(expected_priority_agents)]
    ]
    priority_errors = []
    for index, expected in enumerate(expected_priority_agents):
        recognized = (
            recognized_priority_agents[index]
            if index < len(recognized_priority_agents)
            else None
        )
        if recognized != expected:
            priority_errors.append(
                f"slot {index}: response agent={expected!r}, early recognized={recognized!r}"
            )

    result = {
        "source": source,
        "query": query,
        "response": content,
        "plan": plan,
        "planning_reasoning": reasoning,
        "split_ok": (
            plan is not None
            and not validation_errors
            and not format_errors
        ),
        "split_error": split_error,
        "validation_errors": validation_errors,
        "format_errors": format_errors,
        "priority_match_ok": not priority_errors,
        "priority_errors": priority_errors,
        "recognized_agents": [
            {
                "slot": item.get("slot"),
                "agent": item.get("agent"),
                "recognized_seconds": item.get("recognized_seconds"),
                "confirmed_seconds": item.get("confirmed_seconds"),
                "recognized_step": item.get("recognized_step"),
                "confirmed_step": item.get("confirmed_step"),
                "probability": item.get("probability"),
                "margin": item.get("margin"),
            }
            for item in agent_metrics
            if item.get("agent") is not None
        ],
        "all_agents_recognized_seconds": (
            (metrics.get("agent_priority") or {}).get("all_recognized_seconds")
        ),
        "recognized_agent_count": (
            (metrics.get("agent_priority") or {}).get("recognized_agent_fields")
        ),
        "discovered_agent_count": (
            (metrics.get("agent_priority") or {}).get("discovered_agent_fields")
        ),
        "nfe": metrics.get("nfe"),
        "generated_tokens": metrics.get(
            "generated_tokens", usage.get("completion_tokens")
        ),
        "tps": metrics.get("tps"),
        "raw_fastdllm_metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    tps = result["tps"]
    rendered_tps = f"{tps:.4f}" if tps is not None else "None"
    if result["recognized_agents"]:
        for item in result["recognized_agents"]:
            recognized = item["recognized_seconds"]
            rendered_time = (
                f"{recognized:.4f}s" if recognized is not None else "None"
            )
            print(
                f"agent[{item['slot']}]={item['agent']} "
                f"recognized_time={rendered_time}"
            )
    else:
        print("recognized_agents=none")
    print(
        f"generated_tokens={result['generated_tokens']} "
        f"tps={rendered_tps} nfe={result['nfe']}"
    )


if __name__ == "__main__":
    main()
