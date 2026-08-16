"""Conservative server-side repair for the fixed AOP planner JSON schema."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence, Tuple


PLAN_START = re.compile(r"(?m)^\s*PLAN_JSON\s*:?\s*$")
PLAN_END = re.compile(r"(?m)^\s*END_PLAN_JSON\s*$")
JSON_STRING = r'"(?:\\.|[^"\\])*"'
AGENT_FIELD_START = re.compile(
    r'(?m)(?:^|[\n\[,])\s*\{?\s*"agent"\s*:'
)


def _plan_region(content: str) -> Tuple[int, int, str]:
    start_marker = PLAN_START.search(content)
    if start_marker is None:
        raise ValueError("PLAN_JSON marker is missing")
    end_marker = PLAN_END.search(content, start_marker.end())
    if end_marker is None:
        raise ValueError("END_PLAN_JSON marker is missing")
    array_start = content.find("[", start_marker.end(), end_marker.start())
    if array_start < 0:
        raise ValueError("PLAN_JSON array start is missing")
    return array_start, end_marker.start(), content[array_start:end_marker.start()].strip()


def _strict_plan(text: str) -> List[Dict[str, object]]:
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("PLAN_JSON must be an array of objects")
    return value


def _validate_plan(
    plan: List[Dict[str, object]], allowed_agents: Sequence[str]
) -> None:
    if not plan:
        raise ValueError("PLAN_JSON cannot be empty")
    allowed = set(allowed_agents)
    seen_ids = set()
    for position, item in enumerate(plan, start=1):
        missing = [
            field
            for field in ("agent", "id", "task", "reason", "dep")
            if field not in item
        ]
        if missing:
            raise ValueError(f"plan item {position} is missing fields {missing}")
        if item["agent"] not in allowed:
            raise ValueError(
                f"plan item {position} has unsupported agent {item['agent']!r}"
            )
        item_id = item["id"]
        if not isinstance(item_id, int) or isinstance(item_id, bool) or item_id <= 0:
            raise ValueError(f"plan item {position} has invalid id {item_id!r}")
        if item_id in seen_ids:
            raise ValueError(f"plan item {position} has duplicate id {item_id}")
        if not isinstance(item["task"], str) or not item["task"].strip():
            raise ValueError(f"plan item {position} has an empty task")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError(f"plan item {position} has an empty reason")
        dependencies = item["dep"]
        if not isinstance(dependencies, list) or any(
            not isinstance(dep, int) or isinstance(dep, bool)
            for dep in dependencies
        ):
            raise ValueError(f"plan item {position} has invalid dependencies")
        if any(dep not in seen_ids for dep in dependencies):
            raise ValueError(
                f"plan item {position} references a missing or future dependency"
            )
        seen_ids.add(item_id)


def _minimal_syntax_repair(text: str) -> Tuple[str, List[str]]:
    operations = []

    # The most frequent LLaDA corruption is a new object starting immediately
    # after the previous final dep field, with both the closing brace and comma
    # omitted:  "dep": [1]\n  { ...
    repaired, count = re.subn(
        r'("dep"\s*:\s*\[[^\]\r\n]*\])(\s*)(?=\{)',
        lambda match: match.group(1) + "\n  }," + match.group(2),
        text,
    )
    if count:
        operations.append(f"close_object_and_add_comma:{count}")

    # Another observed form closes the previous object but omits the opening
    # brace before the next agent-first item.
    repaired, count = re.subn(
        r'(\}\s*,)(\s*)(?="agent"\s*:)',
        lambda match: match.group(1) + match.group(2) + "{",
        repaired,
    )
    if count:
        operations.append(f"open_next_object:{count}")

    # Observed lexical corruption:  "agent": "code_agent" ,id": 3
    repaired, count = re.subn(
        r'([,{]\s*)id"\s*:',
        r'\1"id":',
        repaired,
    )
    if count:
        operations.append(f"restore_id_open_quote:{count}")

    return repaired, operations


def _extract_string_field(chunk: str, field: str) -> str:
    match = re.search(
        rf'"{re.escape(field)}"\s*:\s*({JSON_STRING})', chunk
    )
    if match is None:
        raise ValueError(f"missing or malformed {field} field")
    value = json.loads(match.group(1))
    if not isinstance(value, str):
        raise ValueError(f"{field} is not a string")
    return value


def _schema_rebuild(text: str) -> List[Dict[str, object]]:
    starts = list(AGENT_FIELD_START.finditer(text))
    if not starts:
        raise ValueError("no agent-first plan objects were found")
    plan = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        chunk = text[match.start():end]
        id_match = re.search(r'"id"\s*:\s*(\d+)', chunk)
        dep_match = re.search(r'"dep"\s*:\s*(\[[^\]]*\])', chunk)
        if id_match is None:
            raise ValueError(f"object {index + 1} has no integer id")
        if dep_match is None:
            raise ValueError(f"object {index + 1} has no dependency array")
        dependencies = json.loads(dep_match.group(1))
        plan.append(
            {
                "agent": _extract_string_field(chunk, "agent"),
                "id": int(id_match.group(1)),
                "task": _extract_string_field(chunk, "task"),
                "reason": _extract_string_field(chunk, "reason"),
                "dep": dependencies,
            }
        )
    return plan


def _replace_plan_region(content: str, start: int, end: int, plan: object) -> str:
    rendered = json.dumps(plan, ensure_ascii=False, indent=2)
    return content[:start] + rendered + "\n" + content[end:]


def repair_plan_json_response(
    content: str,
    allowed_agents: Sequence[str],
) -> Tuple[str, Dict[str, object]]:
    """Return a strictly parseable response plus an auditable repair report."""

    report: Dict[str, object] = {
        "applied": False,
        "method": "strict",
        "operations": [],
        "original_error": None,
        "repair_error": None,
    }
    start = None
    end = None
    plan_text = ""
    try:
        start, end, plan_text = _plan_region(content)
        strict = _strict_plan(plan_text)
        _validate_plan(strict, allowed_agents)
        return content, report
    except Exception as error:
        report["original_error"] = str(error)
    if start is None or end is None:
        report.update(
            {
                "method": "failed",
                "repair_error": "PLAN_JSON region is unavailable",
            }
        )
        return content, report

    try:
        repaired_text, operations = _minimal_syntax_repair(plan_text)
        repaired = _strict_plan(repaired_text)
        _validate_plan(repaired, allowed_agents)
        report.update(
            {
                "applied": True,
                "method": "minimal_syntax",
                "operations": operations,
            }
        )
        return _replace_plan_region(content, start, end, repaired), report
    except Exception as minimal_error:
        report["minimal_repair_error"] = str(minimal_error)

    try:
        rebuilt = _schema_rebuild(repaired_text)
        _validate_plan(rebuilt, allowed_agents)
        report.update(
            {
                "applied": True,
                "method": "schema_rebuild",
                "operations": operations + ["rebuild_from_validated_fields"],
            }
        )
        return _replace_plan_region(content, start, end, rebuilt), report
    except Exception as repair_error:
        report.update(
            {
                "method": "failed",
                "operations": operations,
                "repair_error": str(repair_error),
            }
        )
        return content, report
