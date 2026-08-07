"""OpenAI-compatible HTTP server for Fast-dLLM v1 LLaDA inference.

The LLaDA decoder produces a complete masked-diffusion result rather than stable
autoregressive tokens, so this server intentionally rejects ``stream=true``.
"""

import argparse
import asyncio
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from generate import generate, generate_with_dual_cache, generate_with_prefix_cache
from model.modeling_llada import LLaDAModelLM


# Default to the original total diffusion-step count. The server CLI can
# override this independently of the requested generation length.
TOTAL_STEPS = 128
MODEL_PATH = os.getenv("FASTDLLM_MODEL_PATH", "/data/labshare/Param/llada")
SERVED_MODEL_NAME = os.getenv("FASTDLLM_SERVED_MODEL_NAME", MODEL_PATH)
HOST = os.getenv("FASTDLLM_HOST", "0.0.0.0")
PORT = int(os.getenv("FASTDLLM_PORT", "7004"))
DEVICE = os.getenv("FASTDLLM_DEVICE", "cuda")
DTYPE = os.getenv("FASTDLLM_DTYPE", "bfloat16")
API_KEY = os.getenv("FASTDLLM_API_KEY")
LOG_LEVEL = os.getenv("FASTDLLM_LOG_LEVEL", "info")


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant"]
    content: Optional[Union[str, List[TextContentPart]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


@dataclass
class ServerConfig:
    model_path: str
    served_model_name: str
    device: str
    dtype: str
    cache_mode: str
    block_size: int
    gen_length: int
    steps: int
    threshold: Optional[float]
    api_key: Optional[str]


class LLaDARuntime:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.lock: Optional[asyncio.Lock] = None

    def _torch_dtype(self):
        if self.config.dtype == "auto":
            return torch.bfloat16 if self.device.type == "cuda" else torch.float32
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.dtype]

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.model = LLaDAModelLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=self._torch_dtype(),
        ).to(self.device).eval()
        self.lock = asyncio.Lock()

    @staticmethod
    def _content_to_text(message: ChatMessage) -> str:
        if message.content is None:
            return ""
        if isinstance(message.content, str):
            return message.content
        return "\n".join(part.text for part in message.content)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        normalized = [
            {"role": message.role, "content": self._content_to_text(message)}
            for message in messages
        ]
        if not any(message["role"] == "user" for message in normalized):
            raise ValueError("At least one user message is required.")
        return normalized

    def _effective_lengths(self, requested_tokens: Optional[int]) -> Tuple[int, int, int]:
        visible_tokens = requested_tokens or self.config.gen_length
        if visible_tokens > self.config.gen_length:
            raise ValueError(
                f"Requested max_tokens={visible_tokens} exceeds the server limit "
                f"of {self.config.gen_length}."
            )
        gen_length = max(
            self.config.block_size,
            math.ceil(visible_tokens / self.config.block_size) * self.config.block_size,
        )
        num_blocks = gen_length // self.config.block_size
        if self.config.steps % num_blocks != 0:
            raise ValueError(
                f"The effective gen_length={gen_length} creates {num_blocks} "
                f"blocks, but configured steps={self.config.steps} is not "
                "divisible by that block count. Choose max_tokens and block_size "
                "that produce a divisor of steps."
            )
        return visible_tokens, gen_length, self.config.steps

    @staticmethod
    def _apply_stop(
        text: str,
        stop: Optional[Union[str, List[str]]],
    ) -> Tuple[str, bool]:
        if not stop:
            return text, False
        stop_values = [stop] if isinstance(stop, str) else stop
        positions = [text.find(value) for value in stop_values if value and value in text]
        if not positions:
            return text, False
        return text[:min(positions)], True

    def generate(self, request: ChatCompletionRequest):
        if request.model != self.config.served_model_name:
            raise ValueError(
                f"Model {request.model!r} is not served. Use "
                f"{self.config.served_model_name!r}."
            )
        if request.n != 1:
            raise ValueError("Fast-dLLM v1 currently supports only n=1.")
        if request.top_p != 1.0:
            raise ValueError("top_p sampling is not supported; use top_p=1.")

        requested_tokens = request.max_completion_tokens or request.max_tokens
        visible_tokens, gen_length, steps = self._effective_lengths(requested_tokens)
        messages = self._prepare_messages(request.messages)
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)

        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)

        generate_fn = {
            "none": generate,
            "prefix": generate_with_prefix_cache,
            "dual": generate_with_dual_cache,
        }[self.config.cache_mode]
        mask_id = self.tokenizer.mask_token_id or 126336
        torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids, nfe = generate_fn(
                model=self.model,
                prompt=input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=self.config.block_size,
                temperature=request.temperature,
                remasking="low_confidence",
                mask_id=mask_id,
                threshold=self.config.threshold,
            )
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started_at

        suffix_ids = output_ids[0, input_ids.shape[1]:input_ids.shape[1] + visible_tokens]
        token_ids = suffix_ids.tolist()
        if mask_id in token_ids:
            raise RuntimeError(
                "Generation ended with unresolved mask tokens. Check the "
                "gen-length, block-size, and cache-mode combination."
            )

        finish_reason = "length"
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id in token_ids:
            token_ids = token_ids[:token_ids.index(eos_token_id)]
            finish_reason = "stop"

        content = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        content, stopped = self._apply_stop(content, request.stop)
        if stopped:
            finish_reason = "stop"

        completion_tokens = len(
            self.tokenizer(content, add_special_tokens=False).input_ids
        )
        prompt_tokens = int(input_ids.shape[1])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        metrics = {
            "nfe": int(nfe),
            "generation_time": elapsed,

            # 固定生成槽位速度
            "slot_tps": visible_tokens / elapsed if elapsed > 0 else 0.0,

            # 实际返回文本速度
            "useful_tps": completion_tokens / elapsed if elapsed > 0 else 0.0,

            "generated_slots": visible_tokens,
            "useful_tokens": completion_tokens,
            "steps": steps,
            "block_size": self.config.block_size,
            "cache_mode": self.config.cache_mode,
            "threshold": self.config.threshold,
        }
        return content, finish_reason, usage, metrics


runtime: Optional[LLaDARuntime] = None


def get_runtime() -> LLaDARuntime:
    if runtime is None:
        raise RuntimeError("The server runtime has not been configured.")
    return runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    current_runtime = get_runtime()
    current_runtime.load()
    yield


app = FastAPI(
    title="Fast-dLLM v1 LLaDA OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)


def openai_error(message: str, error_type: str = "invalid_request_error") -> Dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def check_authorization(request: Request) -> None:
    config = get_runtime().config
    if config.api_key is None:
        return
    if request.headers.get("authorization") != f"Bearer {config.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error(str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=openai_error(str(exc)),
    )


@app.get("/health")
async def health():
    current_runtime = get_runtime()
    return {
        "status": "ok",
        "model_loaded": current_runtime.model is not None,
        "model": current_runtime.config.served_model_name,
    }


def model_description() -> Dict[str, Any]:
    config = get_runtime().config
    return {
        "id": config.served_model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "fast-dllm",
    }


@app.get("/v1/models")
async def list_models(request: Request):
    check_authorization(request)
    return {"object": "list", "data": [model_description()]}


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    check_authorization(request)
    model = model_description()
    if model_id != model["id"]:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} was not found.")
    return model


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    check_authorization(request)
    if payload.stream:
        raise HTTPException(
            status_code=400,
            detail=(
                "stream=true is not supported because LLaDA masked-diffusion "
                "decoding does not produce stable incremental tokens."
            ),
        )
    current_runtime = get_runtime()
    if current_runtime.lock is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        async with current_runtime.lock:
            content, finish_reason, usage, metrics = await asyncio.to_thread(
                current_runtime.generate,
                payload,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "fastdllm": metrics,
    }


def parse_optional_float(value: str) -> Optional[float]:
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threshold must be a number between 0 and 1, or 'none'."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Fast-dLLM v1 LLaDA through an OpenAI-compatible API."
    )
    parser.add_argument(
        "--cache-mode",
        choices=("none", "prefix", "dual"),
        default="dual",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument(
        "--steps",
        type=int,
        default=TOTAL_STEPS,
        help="Total diffusion steps per request, independent of gen-length.",
    )
    parser.add_argument(
        "--threshold",
        type=parse_optional_float,
        default=0.9,
        help=(
            "Confidence threshold for unmasking (0 to 1). Use 'none' to "
            "restore quota-based token transfer."
        ),
    )
    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> None:
    if args.block_size <= 0:
        raise ValueError("block-size must be greater than zero.")
    if args.gen_length <= 0:
        raise ValueError("gen-length must be greater than zero.")
    if args.steps <= 0:
        raise ValueError("steps must be greater than zero.")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1, or 'none'.")
    if args.gen_length % args.block_size != 0:
        raise ValueError("gen-length must be divisible by block-size.")
    num_blocks = args.gen_length // args.block_size
    if args.steps % num_blocks != 0:
        raise ValueError(
            f"gen-length/block-size creates {num_blocks} blocks, but "
            f"steps={args.steps} must be divisible by the block count."
        )


def main() -> None:
    global runtime
    args = parse_args()
    validate_config(args)
    runtime = LLaDARuntime(
        ServerConfig(
            model_path=MODEL_PATH,
            served_model_name=SERVED_MODEL_NAME,
            device=DEVICE,
            dtype=DTYPE,
            cache_mode=args.cache_mode,
            block_size=args.block_size,
            gen_length=args.gen_length,
            steps=args.steps,
            threshold=args.threshold,
            api_key=API_KEY,
        )
    )
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
    )


if __name__ == "__main__":
    main()
