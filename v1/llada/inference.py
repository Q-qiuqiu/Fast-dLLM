"""One-shot multi-Agent planner using LLaDA agent-name-first decoding."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from agent_priority import (
    AgentDecodingController,
    AgentEventType,
    AgentPriorityConfig,
    AgentSpec,
    catalog_from_dicts,
    configure_agent_file_logging,
)
from generate import generate, generate_with_dual_cache, generate_with_prefix_cache
from step_trace import StepTraceWriter


DEFAULT_CATALOG = [
    AgentSpec("search_agent", cold_start_seconds=3.0, wrong_preload_cost_seconds=0.4),
    AgentSpec("code_agent", cold_start_seconds=5.0, wrong_preload_cost_seconds=0.8),
    AgentSpec("summary_agent", cold_start_seconds=2.0, wrong_preload_cost_seconds=0.2),
]

DEFAULT_TASK = """分析 Fast-dLLM v1 的 LLaDA 解码实现，检索扩散模型加速方法，修改代码并总结结果。请拆成最多四个可并行子任务。"""


def load_catalog(path):
    if path is None:
        return DEFAULT_CATALOG
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Agent Catalog JSON must be a list of objects.")
    return catalog_from_dicts(payload)


def planner_prompt(tokenizer, task, catalog):
    names = ", ".join(spec.name for spec in catalog)
    content = (
        f"{task}\n\n"
        "You are a multi-agent planner. Select at most four agents. The decoder "
        "owns a compact internal layout: <agents> contains four catalog names "
        "separated by |, then <task0> through <task3> contain their corresponding "
        "task descriptions. Use the exact name 'none' and task 'none' for unused "
        f"slots. Never invent an agent. Registered agents: {names}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def render_policy_prompt(tokenizer, query, catalog, policy):
    if policy == "raw":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return planner_prompt(tokenizer, query, catalog)


def simulated_preload(event):
    """Example callback; this runs on the dispatcher's daemon worker."""

    if event.event_type == AgentEventType.PRELOAD_START:
        logging.getLogger("fastdllm.agent_priority.loader").info(
            "simulate load %s", event.agent_name
        )
    elif event.event_type == AgentEventType.PRELOAD_CANCEL:
        logging.getLogger("fastdllm.agent_priority.loader").info(
            "simulate cancel %s", event.previous_agent_name
        )
    elif event.event_type == AgentEventType.PRELOAD_SWITCH:
        logging.getLogger("fastdllm.agent_priority.loader").info(
            "simulate switch %s -> %s",
            event.previous_agent_name,
            event.agent_name,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data/labshare/Param/llada")
    parser.add_argument("--catalog", default="config/agent_catalog.example.json", help="Path to an Agent Catalog JSON file.")
    parser.add_argument("--query", "--task", dest="query", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--cache-mode", choices=("none", "prefix", "dual"), default="dual")
    parser.add_argument(
        "--policy",
        choices=("raw", "mid", "planreason"),
        default="planreason",
        help=(
            "raw: query prompt + original decoder; mid: planner prompt + "
            "original decoder; planreason: planner prompt + Agent-priority "
            "decoder (formerly now)."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--agent-log-path",
        default=None,
        help=(
            "Optional file for verbose per-step Agent logs (rotates at 20 MiB). "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--save-step-trace",
        action="store_true",
        help="Save the generated suffix after every diffusion step.",
    )
    parser.add_argument(
        "--step-trace-path",
        default="decode_trace.jsonl",
        help="JSONL output used with --save-step-trace.",
    )
    return parser.parse_args(argv)


def main():
    from transformers import AutoTokenizer
    from model.modeling_llada import LLaDAModelLM

    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    agent_log_path = configure_agent_file_logging(
        args.agent_log_path,
        level=getattr(logging, args.log_level.upper()),
    )
    if args.gen_length % args.block_size:
        raise ValueError("gen-length must be divisible by block-size.")
    blocks = args.gen_length // args.block_size
    if args.steps % blocks:
        raise ValueError("steps must be divisible by the number of blocks.")

    catalog = load_catalog(args.catalog) if args.policy != "raw" else DEFAULT_CATALOG
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rendered = render_policy_prompt(tokenizer, args.query, catalog, args.policy)
    prompt = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    mask_id = tokenizer.mask_token_id or 126336

    controller = None
    if args.policy == "planreason":
        controller = AgentDecodingController(
            tokenizer=tokenizer,
            config=AgentPriorityConfig(catalog=catalog, slots=4),
            prompt_length=prompt.shape[1],
            gen_length=args.gen_length,
            block_length=args.block_size,
            total_steps=args.steps,
            mask_id=mask_id,
            event_callback=simulated_preload,
        )
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    generate_fn = {
        "none": generate,
        "prefix": generate_with_prefix_cache,
        "dual": generate_with_dual_cache,
    }[args.cache_mode]
    trace_writer = None
    if args.save_step_trace:
        trace_writer = StepTraceWriter(
            tokenizer=tokenizer,
            prompt_length=prompt.shape[1],
            mask_id=mask_id,
            path=args.step_trace_path,
        )
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output, nfe = generate_fn(
                model=model,
                prompt=prompt,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_size,
                temperature=0.0,
                remasking="low_confidence",
                mask_id=mask_id,
                threshold=None,
                agent_controller=controller,
                step_callback=trace_writer,
            )
    finally:
        if trace_writer is not None:
            trace_writer.close()
    elapsed = time.perf_counter() - started

    if args.policy == "planreason":
        print(controller.plan(output).render())
        controller.dispatcher.drain()
    else:
        suffix = output[0, prompt.shape[1] :]
        print(tokenizer.decode(suffix, skip_special_tokens=True))
    if controller is not None:
        controller.dispatcher.close(wait=True)
    logging.getLogger("fastdllm.agent_priority").info(
        "generation_summary policy=%s cache_mode=%s nfe=%d elapsed_seconds=%.3f log_path=%s",
        args.policy,
        args.cache_mode,
        nfe,
        elapsed,
        agent_log_path,
    )


if __name__ == "__main__":
    main()
