# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""One-shot LLaDA planner inference with a first-block MASK_SLOT route.

Edit ``PLANNER_TASK`` and ``AGENT_NAMES`` below to change the static request.
Unlike ``chat.py``, this script performs exactly one generation and exits.
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from generate import (
    build_agent_route_template,
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
    max_agent_slots_for_block,
)
from model.modeling_llada import LLaDAModelLM


# ---------------------------------------------------------------------------
# Static planner input. Edit these constants for your own multi-agent scenario.
# IDs in the generated route are 1-based in the same order as AGENT_NAMES;
# ID 0 means that a preallocated slot is unused.
# ---------------------------------------------------------------------------
AGENT_NAMES = [
    "researcher",
    "coder",
    "reviewer",
    "writer",
    "tool_agent",
]

PLANNER_TASK = """
分析 Fast-dLLM v1 中 LLaDA 的推理加速实现，并完成以下目标：
1. 说明 block-wise diffusion、Prefix Cache 和 Dual Cache 的区别；
2. 检查当前 routing MASK_SLOT 解码策略可能存在的问题；
3. 给出可以落地的代码修改建议；
4. 最后汇总为一份结构清晰的技术报告。

请把任务拆成可以并行执行的子任务，并为每个子任务选择最合适的 agent。
""".strip()

MASK_ID = 126336


class StepTraceRecorder:
    """Record only the human-readable partial response after every forward pass."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.steps = []

    def decode_with_masks(self, token_ids):
        pieces = []
        decoded_run = []

        def flush_decoded_run():
            if decoded_run:
                pieces.append(
                    self.tokenizer.decode(decoded_run, skip_special_tokens=False)
                )
                decoded_run.clear()

        for token_id in token_ids:
            if token_id == MASK_ID:
                flush_decoded_run()
                pieces.append("MASK")
            else:
                decoded_run.append(token_id)
        flush_decoded_run()
        return "".join(pieces)

    def __call__(self, nfe, block_index, block_step, state):
        suffix_ids = state[0, self.prompt_length:].detach().cpu()
        self.steps.append(
            {
                "step": nfe,
                "content": self.decode_with_masks(suffix_ids.tolist()),
            }
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.steps, file, ensure_ascii=False, indent=2)


def build_static_prompt(tokenizer, agent_names):
    registry = ", ".join(
        f"{agent_id}={name}" for agent_id, name in enumerate(agent_names, start=1)
    )
    content = (
        f"{PLANNER_TASK}\n\n"
        "You are a multi-agent planner. The response suffix is already initialized "
        "with <R>n=MASK;a=MASK,...;</R><T>. Fill n with the number of active "
        "agents. Fill every agent MASK_SLOT with exactly one registry ID, using 0 "
        "for unused slots. After <T>, write the detailed subtask assigned to each "
        "selected agent. Do not output IDs outside the registry and do not repeat "
        "the routing header. Agent registry: "
        f"0=unused, {registry}."
    )
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return rendered


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one static multi-agent planner request with LLaDA."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/labshare/Param/llada",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--agent_slots", type=int, default=len(AGENT_NAMES))
    parser.add_argument(
        "--cache_mode",
        choices=("none", "prefix", "dual"),
        default="prefix",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--priority_threshold", type=float, default=0.45)
    parser.add_argument("--priority_margin_threshold", type=float, default=0.20)
    parser.add_argument(
        "--trace_path",
        type=str,
        default="decode_trace.json",
        help="Save each step as human-readable text; unresolved positions are MASK.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_generation_config(args):
    if args.gen_length % args.block_size != 0:
        raise ValueError("gen_length must be divisible by block_size.")
    num_blocks = args.gen_length // args.block_size
    if args.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks.")

    steps_per_block = args.steps // num_blocks
    if args.cache_mode == "dual" and steps_per_block < args.block_size:
        raise ValueError(
            "Dual Cache uses a fixed per-block loop. Set steps >= gen_length so "
            "an unfinished block cannot be skipped."
        )


def main():
    args = parse_args()
    validate_generation_config(args)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    capacity = max_agent_slots_for_block(
        tokenizer=tokenizer,
        block_length=args.block_size,
        agent_names=AGENT_NAMES,
    )
    if args.agent_slots > capacity:
        raise ValueError(
            f"block_size={args.block_size} fits at most {capacity} agent "
            f"MASK_SLOTs for this tokenizer; requested {args.agent_slots}."
        )

    route_template = build_agent_route_template(
        tokenizer=tokenizer,
        gen_length=args.gen_length,
        block_length=args.block_size,
        agent_names=AGENT_NAMES,
        num_agent_slots=args.agent_slots,
        device=device,
    )

    rendered_prompt = build_static_prompt(tokenizer, AGENT_NAMES)
    input_ids = tokenizer(rendered_prompt, return_tensors="pt").input_ids.to(device)
    trace_recorder = StepTraceRecorder(
        tokenizer=tokenizer,
        prompt_length=input_ids.shape[1],
    )

    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    generation_kwargs = {
        "model": model,
        "prompt": input_ids,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_size,
        "temperature": 0.0,
        "remasking": "low_confidence",
        "threshold": args.threshold,
        "suffix_template": route_template.suffix_template,
        "priority_mask": route_template.priority_mask,
        "constraint_ids": route_template.constraint_ids,
        "priority_threshold": args.priority_threshold,
        "priority_margin_threshold": args.priority_margin_threshold,
        "step_callback": trace_recorder,
    }

    generate_fn = {
        "none": generate,
        "prefix": generate_with_prefix_cache,
        "dual": generate_with_dual_cache,
    }[args.cache_mode]

    with torch.inference_mode():
        output_ids, _ = generate_fn(**generation_kwargs)

    suffix_ids = output_ids[:, input_ids.shape[1]:]
    route_count, selected_agents = route_template.parse(suffix_ids)
    task_token_ids = suffix_ids[0, route_template.route_token_length:].tolist()
    if tokenizer.eos_token_id in task_token_ids:
        task_token_ids = task_token_ids[:task_token_ids.index(tokenizer.eos_token_id)]
    task_text = tokenizer.decode(task_token_ids, skip_special_tokens=True).strip()
    task_text = task_text.replace("<T>", "\n").replace("</T>", "\n").strip()

    route_lines = [f"count={route_count}"]
    route_lines.extend(
        f"agent_{index}={agent_name}"
        for index, (_, agent_name) in enumerate(selected_agents)
    )
    formatted_response = (
        "<R>\n"
        + "\n".join(route_lines)
        + "\n</R>\n<T>\n"
        + task_text
        + "\n</T>"
    )

    trace_recorder.save(args.trace_path)
    print(formatted_response)


if __name__ == "__main__":
    main()
