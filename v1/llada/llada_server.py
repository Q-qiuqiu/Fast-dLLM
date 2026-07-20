"""OpenAI-compatible API server for MASK_SLOT-accelerated LLaDA planning."""

import argparse
import asyncio
import json
import math
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from generate import (
    build_agent_route_template,
    generate,
    generate_with_dual_cache,
    generate_with_prefix_cache,
    max_agent_slots_for_block,
)
from model.modeling_llada import LLaDAModelLM


DEFAULT_AGENT_NAMES = [
    "researcher",
    "coder",
    "reviewer",
    "writer",
    "tool_agent",
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
    agent_names: List[str]
    cache_mode: str
    threshold: float
    priority_threshold: float
    priority_margin_threshold: float
    api_key: Optional[str]


class LLaDAPlannerRuntime:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.slot_capacity = 0
        self.lock = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.slot_capacity = max_agent_slots_for_block(
            tokenizer=self.tokenizer,
            block_length=self.config.block_size,
            agent_names=self.config.agent_names,
        )
        if self.config.agent_slots > self.slot_capacity:
            raise ValueError(
                f"block_size={self.config.block_size} fits at most "
                f"{self.slot_capacity} agent MASK_SLOTs; requested "
                f"{self.config.agent_slots}."
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

    def format_response(self, suffix_ids, route_template, visible_tokens, stop):
        route_count, selected_agents = route_template.parse(suffix_ids)
        visible_end = min(visible_tokens, suffix_ids.shape[1])
        task_ids = suffix_ids[0, route_template.route_token_length:visible_end].tolist()
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id in task_ids:
            task_ids = task_ids[:task_ids.index(eos_token_id)]
        task_text = self.tokenizer.decode(task_ids, skip_special_tokens=True).strip()
        task_text = task_text.replace("<T>", "\n").replace("</T>", "\n").strip()
        task_text = self.apply_stop(task_text, stop).strip()

        route_lines = [f"count={route_count}"]
        route_lines.extend(
            f"agent_{index}={agent_name}"
            for index, (_, agent_name) in enumerate(selected_agents)
        )
        return (
            "<R>\n"
            + "\n".join(route_lines)
            + "\n</R>\n<T>\n"
            + task_text
            + "\n</T>"
        )

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
        route_template = build_agent_route_template(
            tokenizer=self.tokenizer,
            gen_length=gen_length,
            block_length=self.config.block_size,
            agent_names=self.config.agent_names,
            num_agent_slots=self.config.agent_slots,
            device=self.device,
        )
        if visible_tokens < route_template.route_token_length:
            raise ValueError(
                f"max_tokens must be at least {route_template.route_token_length} "
                "to contain the routing header."
            )

        messages = self.prepare_messages(request.messages)
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)

        generation_kwargs = {
            "model": self.model,
            "prompt": input_ids,
            "steps": steps,
            "gen_length": gen_length,
            "block_length": self.config.block_size,
            "temperature": request.temperature,
            "remasking": "low_confidence",
            "threshold": self.config.threshold,
            "suffix_template": route_template.suffix_template,
            "priority_mask": route_template.priority_mask,
            "constraint_ids": route_template.constraint_ids,
            "priority_threshold": self.config.priority_threshold,
            "priority_margin_threshold": self.config.priority_margin_threshold,
        }
        generate_fn = {
            "none": generate,
            "prefix": generate_with_prefix_cache,
            "dual": generate_with_dual_cache,
        }[self.config.cache_mode]

        with torch.inference_mode():
            output_ids, _ = generate_fn(**generation_kwargs)
        suffix_ids = output_ids[:, input_ids.shape[1]:]
        content = self.format_response(
            suffix_ids,
            route_template,
            visible_tokens,
            request.stop,
        )
        visible_suffix_ids = suffix_ids[0, :min(visible_tokens, suffix_ids.shape[1])].tolist()
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id in visible_suffix_ids:
            completion_tokens = visible_suffix_ids.index(eos_token_id) + 1
        else:
            completion_tokens = len(visible_suffix_ids)
        usage = {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(input_ids.shape[1] + completion_tokens),
        }
        return content, usage


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
            content, usage = await asyncio.to_thread(runtime.generate, payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

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
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve LLaDA with an OpenAI API.")
    parser.add_argument("--model_path", default="/data/labshare/Param/llada")
    parser.add_argument("--served_model_name", default="fast-dllm-llada-planner")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--max_gen_length", type=int, default=256)
    parser.add_argument("--steps_per_block", type=int, default=32)
    parser.add_argument("--agent_slots", type=int, default=5)
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
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--log_level", default="info")
    return parser.parse_args()


def main():
    global runtime
    args = parse_args()
    agent_names = [name.strip() for name in args.agent_names.split(",") if name.strip()]
    if not agent_names:
        raise ValueError("agent_names cannot be empty.")
    if args.max_gen_length % args.block_size != 0:
        raise ValueError("max_gen_length must be divisible by block_size.")

    runtime = LLaDAPlannerRuntime(
        ServerConfig(
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            device=args.device,
            block_size=args.block_size,
            max_gen_length=args.max_gen_length,
            steps_per_block=args.steps_per_block,
            agent_slots=args.agent_slots,
            agent_names=agent_names,
            cache_mode=args.cache_mode,
            threshold=args.threshold,
            priority_threshold=args.priority_threshold,
            priority_margin_threshold=args.priority_margin_threshold,
            api_key=args.api_key,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
