import time
import uuid
from typing import List, Optional, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "/data/labshare/Param/Fast_dLLM_v2_7B"

app = FastAPI(title="Fast-dLLM Chat API", version="1.0")


def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    messages: List[ChatMessage]
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    block_size: int = Field(default=32, ge=1)
    small_block_size: int = Field(default=8, ge=1)
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    seed: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list
    usage: dict


@torch.no_grad()
def generate_reply(messages: List[ChatMessage], req: ChatCompletionRequest) -> str:
    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs["input_ids"],
        tokenizer=tokenizer,
        block_size=req.block_size,
        max_new_tokens=req.max_tokens,
        small_block_size=req.small_block_size,
        threshold=req.threshold,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    prompt_len = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][prompt_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


@app.on_event("startup")
def load_model() -> None:
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@app.post("/v1/chat/competitons", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    if req.stream:
        raise HTTPException(status_code=400, detail="stream is not supported")
    if req.seed is not None:
        fix_seed(req.seed)

    response_text = generate_reply(req.messages, req)

    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
    completion_tokens = len(tokenizer(response_text, add_special_tokens=False).input_ids)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model or MODEL_NAME,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7001)
