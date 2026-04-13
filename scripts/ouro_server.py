"""
Ouro-2.6B OpenAI-compatible inference server.

Serves ByteDance/Ouro-2.6B on Apple Silicon MPS via a lightweight
FastAPI server that exposes /v1/chat/completions and /v1/models.

Usage:
    source ~/.venv-vllm-metal/bin/activate
    python scripts/ouro_server.py [--port 8100] [--model ByteDance/Ouro-2.6B]

Requires transformers==4.54.1 (recommended by ByteDance for Ouro).
"""

import argparse
import time
import uuid
import logging

import torch

# ------------------------------------------------------------------
# Compatibility patches for Ouro + transformers 4.54.x
#
# Ouro's UniversalTransformerCache stores KV data in key_cache /
# value_cache lists (old API), but transformers 4.54 deprecated
# those as @property wrappers over a new layers-based API.
# Two targeted patches make them coexist:
#
#   1. Remove the deprecated property descriptors so __init__ can
#      assign key_cache / value_cache as plain lists.
#
#   2. Patch Cache.get_mask_sizes() to derive kv_length from
#      key_cache (or cache_position on prefill) instead of the
#      layers list that Ouro's cache doesn't populate.
# ------------------------------------------------------------------
from transformers.cache_utils import Cache

for _attr in ("key_cache", "value_cache"):
    if isinstance(Cache.__dict__.get(_attr), property):
        delattr(Cache, _attr)

_orig_get_mask_sizes = getattr(Cache, "get_mask_sizes", None)


def _patched_get_mask_sizes(self, cache_position, layer_idx=0):
    if self.layers and layer_idx < len(self.layers):
        return _orig_get_mask_sizes(self, cache_position, layer_idx)

    kc = getattr(self, "key_cache", None)
    if (
        kc
        and isinstance(kc, list)
        and layer_idx < len(kc)
        and kc[layer_idx] is not None
    ):
        return kc[layer_idx].shape[2], 0

    if cache_position is not None and cache_position.numel() > 0:
        return int(cache_position[-1].item()) + 1, 0

    return 0, 0


if _orig_get_mask_sizes is not None:
    Cache.get_mask_sizes = _patched_get_mask_sizes

# ------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field

logger = logging.getLogger("ouro_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(title="Ouro Inference Server")

_model = None
_tokenizer = None
_model_id = None
_device = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: list[str] | None = None


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_id,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_id, "device": str(_device)}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    t0 = time.time()

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    n_input = inputs["input_ids"].shape[1]

    do_sample = req.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=req.max_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p

    with torch.no_grad():
        output = _model.generate(**inputs, **gen_kwargs)

    n_gen = output.shape[1] - n_input
    text = _tokenizer.decode(output[0][n_input:], skip_special_tokens=True)
    elapsed = time.time() - t0
    tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

    logger.info(
        f"Generated {n_gen} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)"
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": n_input,
            "completion_tokens": n_gen,
            "total_tokens": n_input + n_gen,
        },
    }


def load_model(model_path: str):
    global _model, _tokenizer, _model_id, _device

    _model_id = model_path
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading {model_path} on {_device}...")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=_device,
    )
    _model.eval()

    logger.info(
        f"Model loaded. Parameters: "
        f"{sum(p.numel() for p in _model.parameters()) / 1e9:.1f}B"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ouro OpenAI-compatible server"
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model", type=str, default="ByteDance/Ouro-2.6B")
    args = parser.parse_args()

    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
