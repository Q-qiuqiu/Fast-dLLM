"""OpenAI-compatible API server for MASK_SLOT-accelerated LLaDA planning."""

import argparse
import asyncio
import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from agent_priority import configure_agent_file_logging
from agent_timing import AgentTimingRecorder
from generate import generate, generate_with_dual_cache, generate_with_prefix_cache
from json_agent_priority import (
    JsonAgentFieldController,
    JsonAgentPriorityConfig,
    extract_agent_registry,
)
from model.modeling_llada import LLaDAModelLM
from planner_json_repair import repair_plan_json_response


DEFAULT_AGENT_NAMES = [
    "code_agent",
    "math_agent",
    "search_agent",
    "commonsense_agent",
]


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stop: Optional[Union[str, List[str]]] = None


@dataclass
class ServerConfig:
    model_path: str
    served_model_name: str
    device: str
    block_size: int
    max_gen_length: int
    steps_per_block: int
    agent_slots: int
    agent_timing_slots: int
    agent_names: List[str]
    cache_mode: str
    threshold: float
    priority_threshold: float
    priority_margin_threshold: float
    agent_anchor_margin: float
    agent_discovery_steps: int
    agent_timing_log_path: str
    agent_timing_summary_path: Optional[str]
    plan_json_repair: bool
    policy: str
    api_key: Optional[str]


class LLaDAPlannerRuntime:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.lock = None
        self.timing_recorder = AgentTimingRecorder(
            config.agent_timing_log_path,
            config.agent_timing_summary_path,
        )

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.model = LLaDAModelLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        self.lock = asyncio.Lock()

    @staticmethod
    def message_content_to_text(message: ChatMessage) -> str:
        if message.content is None:
            return ""
        if isinstance(message.content, str):
            return message.content
        text_parts = []
        for part in message.content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            else:
                raise ValueError(
                    f"Unsupported message content part {part.type!r}; only text is supported."
                )
        return "\n".join(text_parts)

    def prepare_messages(self, messages: List[ChatMessage]):
        normalized = []
        has_user_message = False
        for message in messages:
            if message.role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported role {message.role!r}; expected system, user, or assistant."
                )
            normalized.append(
                {
                    "role": message.role,
                    "content": self.message_content_to_text(message),
                }
            )
            if message.role == "user":
                has_user_message = True

        if not has_user_message:
            raise ValueError("At least one user message is required.")
        return normalized

    def record_agent_timing(
        self,
        *,
        completion_id: str,
        created: int,
        request: ChatCompletionRequest,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        query = next(
            (
                self.message_content_to_text(message)
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            "",
        )
        requested_tokens = request.max_completion_tokens or request.max_tokens
        try:
            self.timing_recorder.record(
                completion_id=completion_id,
                created_unix=created,
                query=query,
                model=request.model,
                temperature=request.temperature,
                requested_max_tokens=requested_tokens,
                metrics=metrics,
                error=error,
            )
        except Exception:
            # A late filesystem failure must be visible in the server log, but
            # must not discard an otherwise successful benchmark response.
            logging.getLogger("fastdllm.agent_timing").exception(
                "Failed to persist Agent timing for %s", completion_id
            )

    def effective_lengths(self, requested_tokens: Optional[int]):
        visible_tokens = requested_tokens or self.config.max_gen_length
        if visible_tokens > self.config.max_gen_length:
            raise ValueError(
                f"Requested max_tokens={visible_tokens} exceeds server limit "
                f"{self.config.max_gen_length}."
            )
        gen_length = max(
            self.config.block_size,
            math.ceil(visible_tokens / self.config.block_size) * self.config.block_size,
        )
        num_blocks = gen_length // self.config.block_size
        steps = num_blocks * self.config.steps_per_block
        if self.config.cache_mode == "dual" and self.config.steps_per_block < self.config.block_size:
            raise ValueError(
                "Dual Cache requires steps_per_block >= block_size so an unfinished "
                "block cannot be skipped."
            )
        return visible_tokens, gen_length, steps

    @staticmethod
    def apply_stop(text: str, stop: Optional[Union[str, List[str]]]):
        if not stop:
            return text
        stops = [stop] if isinstance(stop, str) else stop
        indices = [text.find(value) for value in stops if value and value in text]
        return text[:min(indices)] if indices else text

    def generate(self, request: ChatCompletionRequest):
        if request.model != self.config.served_model_name:
            raise ValueError(
                f"Model {request.model!r} is not served; use "
                f"{self.config.served_model_name!r}."
            )
        if request.n != 1:
            raise ValueError("Only n=1 is supported.")
        if request.top_p != 1.0:
            raise ValueError("top_p sampling is not supported; use top_p=1.")

        requested_tokens = request.max_completion_tokens or request.max_tokens
        visible_tokens, gen_length, steps = self.effective_lengths(requested_tokens)
        messages = self.prepare_messages(request.messages)
        request_agent_names = extract_agent_registry(
            messages, self.config.agent_names
        )
        if self.config.policy in {"mid", "now"}:
            registry = ", ".join(request_agent_names)
            planner_instruction = {
                "role": "system",
                "content": (
                    "Return a normal planner response with no private routing tags. "
                    "Write PLAN_JSON before PLANNING_REASONING. In PLAN_JSON, use a "
                    "compact JSON array without Markdown fences or indentation. Every "
                    "object must put the agent field first and use this exact prefix: "
                    "{\"agent\":\"<name>\",. Follow it with id, task, reason, and dep. "
                    f"Agent names must be selected from: {registry}. Preserve the "
                    "query's dates and numeric constraints verbatim. Do not create "
                    "duplicate subtasks. Every object must explicitly include reason "
                    "and dep; use dep:[] for independent tasks. Follow the caller's "
                    "Agent capability and routing descriptions exactly. When a "
                    "required external fact is absent from the supplied query or "
                    "context, first create a fact-acquisition subtask using the "
                    "caller's search, retrieval, or context Agent as appropriate. "
                    "Never ask a math, calculation, code, or reasoning Agent to "
                    "discover an absent external fact; computational Agents may use "
                    "only values present in the request or produced by dependencies. "
                    "In particular, a country's population for a named year is an "
                    "external fact when its numeric value is not supplied: create a "
                    "search subtask for each required country before any arithmetic. "
                    "Use exactly one "
                    "standalone marker of each kind, in this order: PLAN_JSON, "
                    "END_PLAN_JSON, PLANNING_REASONING, END_PLANNING_REASONING. "
                    "END_PLANNING_REASONING must be the final output line."
                ),
            }
            first_user = next(
                (index for index, message in enumerate(messages) if message["role"] == "user"),
                len(messages),
            )
            messages.insert(first_user, planner_instruction)
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)
        mask_id = self.tokenizer.mask_token_id or 126336
        controller = None
        if self.config.policy == "now":
            controller = JsonAgentFieldController(
                tokenizer=self.tokenizer,
                config=JsonAgentPriorityConfig(
                    catalog=request_agent_names,
                    priority_slots=self.config.agent_slots,
                    tracking_slots=self.config.agent_timing_slots,
                    anchor_min_logit_margin=self.config.agent_anchor_margin,
                    tentative_probability=self.config.priority_threshold,
                    tentative_margin=self.config.priority_margin_threshold,
                    discovery_steps=self.config.agent_discovery_steps,
                ),
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
            )

        generation_kwargs = {
            "model": self.model,
            "prompt": input_ids,
            "steps": steps,
            "gen_length": gen_length,
            "block_length": self.config.block_size,
            "temperature": request.temperature,
            "remasking": "low_confidence",
            "threshold": self.config.threshold,
            "mask_id": mask_id,
            "agent_controller": controller,
        }
        generate_fn = {
            "none": generate,
            "prefix": generate_with_prefix_cache,
            "dual": generate_with_dual_cache,
        }[self.config.cache_mode]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids, nfe = generate_fn(**generation_kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started_at
        suffix_ids = output_ids[:, input_ids.shape[1]:]
        content = self.tokenizer.decode(
            suffix_ids[0, :visible_tokens], skip_special_tokens=True
        )
        content = self.apply_stop(content, request.stop).strip()
        model_completion_tokens = len(
            self.tokenizer(content, add_special_tokens=False).input_ids
        )
        repair_report = {
            "applied": False,
            "method": "disabled",
            "operations": [],
        }
        if self.config.plan_json_repair and self.config.policy in {"mid", "now"}:
            content, repair_report = repair_plan_json_response(
                content, request_agent_names
            )
        if controller is not None:
            controller.close()
        # OpenAI usage describes the text actually returned to the caller.  A
        # diffusion LM always allocates a fixed output canvas, so counting the
        # canvas (often all 1024 positions) as completion tokens substantially
        # overstates useful throughput when special/padding tokens are skipped.
        completion_tokens = len(
            self.tokenizer(content, add_special_tokens=False).input_ids
        )
        prompt_tokens = int(input_ids.shape[1])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_tokens),
            "total_tokens": prompt_tokens + int(completion_tokens),
        }
        metrics = {
            "nfe": int(nfe),
            "generated_tokens": int(model_completion_tokens),
            "returned_tokens": int(completion_tokens),
            "generation_seconds": generation_seconds,
            "tps": (
                model_completion_tokens / generation_seconds
                if generation_seconds > 0 else 0.0
            ),
            "plan_json_repair": repair_report,
        }
        if controller is not None:
            metrics["agent_priority"] = controller.metrics()
        return content, usage, metrics


runtime: Optional[LLaDAPlannerRuntime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if runtime is None:
        raise RuntimeError("Server runtime was not configured.")
    runtime.load()
    yield


app = FastAPI(title="Fast-dLLM LLaDA OpenAI API", lifespan=lifespan)


def check_authorization(request: Request):
    if runtime.config.api_key is None:
        return
    if request.headers.get("authorization") != f"Bearer {runtime.config.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.exception_handler(HTTPException)
async def openai_http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def openai_validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": runtime.model is not None}


@app.get("/v1/models")
async def list_models(request: Request):
    check_authorization(request)
    return {
        "object": "list",
        "data": [
            {
                "id": runtime.config.served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fast-dllm",
            }
        ],
    }


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    check_authorization(request)
    if model_id != runtime.config.served_model_name:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} was not found.")
    return {
        "id": runtime.config.served_model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "fast-dllm",
    }


def completion_chunk(completion_id, created, model, delta, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    check_authorization(request)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        async with runtime.lock:
            content, usage, metrics = await asyncio.to_thread(runtime.generate, payload)
    except ValueError as error:
        runtime.record_agent_timing(
            completion_id=completion_id,
            created=created,
            request=payload,
            error=str(error),
        )
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        runtime.record_agent_timing(
            completion_id=completion_id,
            created=created,
            request=payload,
            error=str(error),
        )
        raise HTTPException(status_code=500, detail=str(error)) from error

    runtime.record_agent_timing(
        completion_id=completion_id,
        created=created,
        request=payload,
        metrics=metrics,
    )

    if not payload.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "fastdllm": metrics,
        }

    async def stream_response():
        chunks = [
            completion_chunk(
                completion_id, created, payload.model, {"role": "assistant"}
            ),
            completion_chunk(
                completion_id, created, payload.model, {"content": content}
            ),
            completion_chunk(
                completion_id, created, payload.model, {}, finish_reason="stop"
            ),
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        if payload.stream_options and payload.stream_options.get("include_usage"):
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [],
                "usage": usage,
                "fastdllm": metrics,
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve LLaDA with an OpenAI API.")
    parser.add_argument("--model_path", default="/data/labshare/Param/llada")
    parser.add_argument("--served_model_name", default="/data/labshare/Param/llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7004)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--max_gen_length", type=int, default=1024)
    parser.add_argument("--steps_per_block", type=int, default=32)
    parser.add_argument("--agent_slots", type=int, default=4)
    parser.add_argument(
        "--agent_timing_slots",
        type=int,
        default=32,
        help=(
            "Maximum normal-response Agent fields to time. Only agent_slots "
            "fields participate in priority decoding or prefetch."
        ),
    )
    parser.add_argument(
        "--agent_names",
        default=",".join(DEFAULT_AGENT_NAMES),
        help="Comma-separated agent registry in ID order.",
    )
    parser.add_argument(
        "--cache_mode", choices=("none", "prefix", "dual"), default="prefix"
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--priority_threshold", type=float, default=0.45)
    parser.add_argument("--priority_margin_threshold", type=float, default=0.20)
    parser.add_argument(
        "--agent_anchor_margin",
        type=float,
        default=-6.0,
        help="Minimum mean target-vs-top logit margin for a speculative JSON Agent anchor.",
    )
    parser.add_argument(
        "--agent_discovery_steps",
        type=int,
        default=4,
        help=(
            "Compatibility option. JSON Agent discovery now reuses normal "
            "full-sequence block warm-ups and adds no extra model forwards."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("raw", "mid", "now"),
        default="now",
        help="raw=unchanged prompt, mid=JSON planner prompt, now=JSON Agent-priority decoding.",
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--log_level", default="info")
    parser.add_argument(
        "--agent_log_path",
        default="agent_decode.log",
        help="File for verbose Agent step/event logs (rotates at 20 MiB).",
    )
    parser.add_argument(
        "--agent_timing_log_path",
        default="agent_timings.jsonl",
        help=(
            "Append-only JSONL file with one Agent timing record per API request."
        ),
    )
    parser.add_argument(
        "--agent_timing_summary_path",
        default=None,
        help=(
            "Current server-session aggregate JSON. Defaults to "
            "<agent_timing_log_path stem>.summary.json."
        ),
    )
    parser.add_argument(
        "--plan_json_repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Conservatively repair and validate the fixed planner JSON schema "
            "before returning a response. Use --no-plan_json_repair to disable."
        ),
    )
    return parser.parse_args()


def main():
    global runtime
    args = parse_args()
    configure_agent_file_logging(
        args.agent_log_path,
        level=getattr(logging, args.log_level.upper()),
    )
    agent_names = [name.strip() for name in args.agent_names.split(",") if name.strip()]
    if not agent_names:
        raise ValueError("agent_names cannot be empty.")
    if args.max_gen_length % args.block_size != 0:
        raise ValueError("max_gen_length must be divisible by block_size.")
    if args.agent_discovery_steps <= 0:
        raise ValueError("agent_discovery_steps must be positive.")
    if args.agent_timing_slots < args.agent_slots:
        raise ValueError("agent_timing_slots must be at least agent_slots.")

    runtime = LLaDAPlannerRuntime(
        ServerConfig(
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            device=args.device,
            block_size=args.block_size,
            max_gen_length=args.max_gen_length,
            steps_per_block=args.steps_per_block,
            agent_slots=args.agent_slots,
            agent_timing_slots=args.agent_timing_slots,
            agent_names=agent_names,
            cache_mode=args.cache_mode,
            threshold=args.threshold,
            priority_threshold=args.priority_threshold,
            priority_margin_threshold=args.priority_margin_threshold,
            agent_anchor_margin=args.agent_anchor_margin,
            agent_discovery_steps=args.agent_discovery_steps,
            agent_timing_log_path=args.agent_timing_log_path,
            agent_timing_summary_path=args.agent_timing_summary_path,
            plan_json_repair=args.plan_json_repair,
            policy=args.policy,
            api_key=args.api_key,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
