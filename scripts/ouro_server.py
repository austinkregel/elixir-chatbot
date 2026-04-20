"""
Ouro-2.6B OpenAI-compatible inference server.

Serves ByteDance/Ouro-2.6B on Apple Silicon MPS via a lightweight
FastAPI server that exposes /v1/chat/completions and /v1/models.

Implements two key optimizations from the LoopLM paper
("Scaling Latent Reasoning via Looped Language Models"):

  1. Pre-allocated KV cache buffers reused across generate() calls to
     prevent Metal/MPS page accumulation (the primary fix for 250 GB
     memory explosion -- Metal retains freed pages across calls).

  2. Last-step KV cache sharing during decoding (paper section 5.4.2)
     reducing effective cache slots from 192 to 48 with negligible
     quality loss (0.07 point GSM8K, Table 14).

Usage:
    source ~/.venv-vllm-metal/bin/activate
    python scripts/ouro_server.py [--port 8100] [--model ByteDance/Ouro-2.6B]

Environment variables:
    OURO_MAX_RSS_GB    - RSS limit before rejecting requests (default: 12)
    OURO_MAX_SEQ_LEN   - Max sequence length for pre-allocated buffers (default: 16384)
    OURO_UT_STEPS      - Override recurrent UT steps at inference (default: model config)

Requires transformers==4.54.1 (recommended by ByteDance for Ouro).
"""

import argparse
import gc
import os
import sys
import time
import uuid
import logging
import threading

import torch
import psutil

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
#   2. Patch Cache.get_mask_sizes() to derive kv_length from tracked
#      _lengths (for pre-allocated caches), key_cache, or
#      cache_position as fallback.
# ------------------------------------------------------------------
from transformers.cache_utils import Cache

for _attr in ("key_cache", "value_cache"):
    if isinstance(Cache.__dict__.get(_attr), property):
        delattr(Cache, _attr)

_orig_get_mask_sizes = getattr(Cache, "get_mask_sizes", None)

_kv_pool = None  # forward declaration; initialized after model load


def _patched_get_mask_sizes(self, cache_position, layer_idx=0):
    _lengths = getattr(self, "_lengths", None)
    if _lengths is not None and isinstance(_lengths, list):
        # Determine the effective cached length to report.
        length = 0
        if getattr(self, "_preallocated", False) and _kv_pool is not None:
            # With sharing: use the canonical (last UT step's) slot length
            phys_layer = layer_idx % _kv_pool.num_layers
            canonical_idx = (
                (_kv_pool.total_ut_steps - 1) * _kv_pool.num_layers
                + phys_layer
            )
            if canonical_idx < len(_lengths):
                length = _lengths[canonical_idx]
        elif layer_idx < len(_lengths):
            length = _lengths[layer_idx]

        if length > 0:
            return length, 0

        # Pre-allocated with length=0 (empty cache during prefill):
        # skip the kc.shape[2] fallback below (it would return
        # max_seq_len instead of actual content) and go straight
        # to the cache_position fallback.
        if getattr(self, "_preallocated", False):
            if cache_position is not None and cache_position.numel() > 0:
                return int(cache_position[-1].item()) + 1, 0
            return 0, 0

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


# ------------------------------------------------------------------
# MPS Memory Helpers
# ------------------------------------------------------------------

def _mps_allocated_gb() -> float:
    """Bytes currently allocated by PyTorch on MPS, in GB."""
    try:
        return torch.mps.current_allocated_memory() / (1024**3)
    except Exception:
        return -1.0


def _mps_driver_gb() -> float:
    """Total bytes the Metal driver has allocated (includes retained pages)."""
    try:
        return torch.mps.driver_allocated_memory() / (1024**3)
    except Exception:
        return -1.0


# ------------------------------------------------------------------
# Pre-allocated KV Buffer Pool
# ------------------------------------------------------------------

class _KVBufferPool:
    """
    Persistent KV cache buffers that survive across generate() calls.

    Metal/MPS retains page allocations even after PyTorch releases them
    via empty_cache(). By pre-allocating fixed buffers once and reusing
    them for every inference request (via in-place slice assignment
    instead of torch.cat), we prevent the monotonic memory growth that
    causes the 250 GB explosion.
    """

    def __init__(
        self,
        max_slots: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: str,
        num_layers: int,
        total_ut_steps: int,
    ):
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.total_ut_steps = total_ut_steps
        self.dtype = dtype
        self.device = device

        self.key_buffers: list[torch.Tensor] = []
        self.value_buffers: list[torch.Tensor] = []
        for _ in range(max_slots):
            self.key_buffers.append(
                torch.zeros(
                    1, num_kv_heads, max_seq_len, head_dim,
                    dtype=dtype, device=device,
                )
            )
            self.value_buffers.append(
                torch.zeros(
                    1, num_kv_heads, max_seq_len, head_dim,
                    dtype=dtype, device=device,
                )
            )

        total_bytes = sum(
            b.nelement() * b.element_size()
            for b in self.key_buffers + self.value_buffers
        )
        logger.info(
            f"KV buffer pool: {max_slots} slots x {max_seq_len} max_seq "
            f"= {total_bytes / 1e9:.2f} GB on {device}"
        )


# ------------------------------------------------------------------
# UniversalTransformerCache monkey-patches
# ------------------------------------------------------------------

def _patched_ut_init(self, max_cache_size=None):
    """Use pre-allocated buffers from the global pool when available."""
    self._seen_tokens = 0
    self.max_cache_size = max_cache_size
    self.layers = []

    if (
        _kv_pool is not None
        and max_cache_size is not None
        and max_cache_size <= _kv_pool.max_slots
    ):
        self.key_cache = list(_kv_pool.key_buffers[:max_cache_size])
        self.value_cache = list(_kv_pool.value_buffers[:max_cache_size])
        self._lengths = [0] * max_cache_size
        self._preallocated = True
    else:
        self.key_cache = []
        self.value_cache = []
        self._lengths = []
        self._preallocated = False


def _patched_ut_update(self, key_states, value_states, layer_idx,
                       cache_kwargs=None):
    """In-place buffer writes with last-step KV cache sharing (paper 5.4.2)."""
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

    if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
        raise IndexError(
            f"Cache index {layer_idx} exceeds "
            f"max_cache_size={self.max_cache_size}"
        )

    if not getattr(self, "_preallocated", False):
        # --- Fallback: original dynamic torch.cat behavior ---
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self._lengths.append(0)

        cached_key = self.key_cache[layer_idx]
        if cached_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [cached_key, key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )
        result_key = self.key_cache[layer_idx]
        self._seen_tokens = result_key.shape[2]
        self._lengths[layer_idx] = self._seen_tokens
        return result_key, self.value_cache[layer_idx]

    # --- Pre-allocated path with last-step KV cache sharing ---

    # Detect decoding: single new token AND this slot already has data.
    # During prefill (first write to each slot), _lengths[idx] == 0 so
    # this is False even for single-token inputs.
    is_decoding = (
        key_states.shape[2] == 1 and self._lengths[layer_idx] > 0
    )

    if is_decoding and _kv_pool is not None:
        num_layers = _kv_pool.num_layers
        total_ut_steps = _kv_pool.total_ut_steps
        current_ut = layer_idx // num_layers
        phys_layer = layer_idx % num_layers
        last_step = total_ut_steps - 1
        canonical_idx = last_step * num_layers + phys_layer

        if current_ut != last_step:
            # Non-last UT step: read from shared (last step's) cache
            # without writing.  Paper Table 14 "Last-step only" strategy.
            length = self._lengths[canonical_idx]
            return (
                self.key_cache[canonical_idx][:, :, :length, :],
                self.value_cache[canonical_idx][:, :, :length, :],
            )
        # Last UT step: redirect write to the canonical slot
        layer_idx = canonical_idx

    # In-place write into pre-allocated buffer
    pos = self._lengths[layer_idx]
    new_len = key_states.shape[2]
    end = pos + new_len

    if end > _kv_pool.max_seq_len:
        available = _kv_pool.max_seq_len - pos
        if available <= 0:
            if not getattr(_patched_ut_update, "_overflow_logged", False):
                logger.warning(
                    f"Buffer overflow: slot {layer_idx} pos={pos} "
                    f"new_tokens={n_new} max_seq={_kv_pool.max_seq_len} "
                    f"(suppressing further overflow warnings this request)"
                )
                _patched_ut_update._overflow_logged = True
            return (
                self.key_cache[layer_idx][:, :, :pos, :],
                self.value_cache[layer_idx][:, :, :pos, :],
            )
        key_states = key_states[:, :, :available, :]
        value_states = value_states[:, :, :available, :]
        end = pos + available

    self.key_cache[layer_idx][:, :, pos:end, :] = key_states
    self.value_cache[layer_idx][:, :, pos:end, :] = value_states
    self._lengths[layer_idx] = end
    self._seen_tokens = end

    return (
        self.key_cache[layer_idx][:, :, :end, :],
        self.value_cache[layer_idx][:, :, :end, :],
    )


def _patched_ut_get_seq_length(self, layer_idx=0):
    if layer_idx is None:
        layer_idx = 0
    if getattr(self, "_preallocated", False):
        # With sharing, always report the canonical (last step's) slot
        # length so cache_position is computed correctly for all UT steps.
        if _kv_pool is not None:
            phys_layer = layer_idx % _kv_pool.num_layers
            canonical_idx = (
                (_kv_pool.total_ut_steps - 1) * _kv_pool.num_layers
                + phys_layer
            )
            if 0 <= canonical_idx < len(self._lengths):
                return self._lengths[canonical_idx]
        if 0 <= layer_idx < len(self._lengths):
            return self._lengths[layer_idx]
        return 0
    # Original dynamic behavior
    if layer_idx < 0 or len(self.key_cache) <= layer_idx:
        return 0
    cached = self.key_cache[layer_idx]
    if cached is None:
        return 0
    return cached.shape[2]


def _patched_ut_clear(self):
    """Reset lengths without freeing pre-allocated buffers."""
    if getattr(self, "_preallocated", False):
        for i in range(len(self._lengths)):
            self._lengths[i] = 0
        self._seen_tokens = 0
    else:
        self.key_cache = []
        self.value_cache = []
        self._lengths = []
        self._seen_tokens = 0


def _apply_ut_cache_patches():
    """Monkey-patch UniversalTransformerCache after the model is loaded."""
    ut_cache_cls = None
    for name, mod in sys.modules.items():
        if "modeling_ouro" in name:
            cls = getattr(mod, "UniversalTransformerCache", None)
            if cls is not None:
                ut_cache_cls = cls
                break

    if ut_cache_cls is None:
        logger.warning(
            "UniversalTransformerCache not found in loaded modules; "
            "pre-allocated KV cache will not be used"
        )
        return

    ut_cache_cls.__init__ = _patched_ut_init
    ut_cache_cls.update = _patched_ut_update
    ut_cache_cls.get_seq_length = _patched_ut_get_seq_length
    ut_cache_cls.clear = _patched_ut_clear
    logger.info(
        "Patched UniversalTransformerCache for pre-allocated KV buffers "
        "with last-step cache sharing"
    )


# ------------------------------------------------------------------
# Server globals and helpers
# ------------------------------------------------------------------

app = FastAPI(title="Ouro Inference Server")

_model = None
_tokenizer = None
_model_id = None
_device = None
_request_lock = threading.Lock()

MAX_RSS_GB = float(os.environ.get("OURO_MAX_RSS_GB", "12"))


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _force_memory_cleanup():
    """Reclaim GPU and CPU memory."""
    gc.collect()
    if _device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif _device and str(_device).startswith("cuda"):
        torch.cuda.empty_cache()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=10)
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
    rss = round(_rss_gb(), 2)
    mps_alloc = round(_mps_allocated_gb(), 2)
    mps_driver = round(_mps_driver_gb(), 2)
    return {
        "status": "ok",
        "model": _model_id,
        "device": str(_device),
        "rss_gb": rss,
        "mps_allocated_gb": mps_alloc,
        "mps_driver_gb": mps_driver,
        "kv_pool_active": _kv_pool is not None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    rss_before = _rss_gb()
    if rss_before > MAX_RSS_GB:
        logger.warning(
            f"RSS {rss_before:.1f} GB exceeds limit {MAX_RSS_GB} GB, "
            "forcing cleanup before inference"
        )
        _force_memory_cleanup()
        rss_before = _rss_gb()
        if rss_before > MAX_RSS_GB:
            return {
                "error": {
                    "message": f"Memory pressure too high ({rss_before:.1f} GB), try again later",
                    "type": "server_error",
                }
            }

    with _request_lock:
        return _do_generate(req, rss_before)


def _do_generate(req: ChatCompletionRequest, rss_before: float):
    t0 = time.time()
    mps_driver_before = _mps_driver_gb()

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    msg_sizes = [len(m["content"]) for m in messages]
    logger.info(
        f"Request: {len(messages)} messages, "
        f"msg_chars={msg_sizes}, total_chars={sum(msg_sizes)}"
    )
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    n_input = inputs["input_ids"].shape[1]

    n_padding = 0
    if "attention_mask" in inputs:
        n_padding = int((inputs["attention_mask"] == 0).sum().item())
    n_real = n_input - n_padding

    prompt_preview = prompt[:200].replace("\n", "\\n")
    logger.info(
        f"Input shape: {list(inputs['input_ids'].shape)} | "
        f"real_tokens={n_real} padding={n_padding} total={n_input} | "
        f"prompt_chars={len(prompt)} | "
        f"prompt_start: {prompt_preview!r}"
    )

    max_tokens = min(req.max_tokens, 512)
    _patched_ut_update._overflow_logged = False

    do_sample = req.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=_tokenizer.eos_token_id,
        repetition_penalty=req.repetition_penalty,
    )
    if req.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = req.no_repeat_ngram_size
    if do_sample:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p

    output = None
    try:
        with torch.no_grad():
            output = _model.generate(**inputs, **gen_kwargs)

        n_gen = output.shape[1] - n_input
        text = _tokenizer.decode(output[0][n_input:], skip_special_tokens=True)
    finally:
        del inputs
        if output is not None:
            del output
        _force_memory_cleanup()

    elapsed = time.time() - t0
    tok_per_sec = n_gen / elapsed if elapsed > 0 else 0
    rss_after = _rss_gb()
    mps_driver_after = _mps_driver_gb()

    logger.info(
        f"Generated {n_gen} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s) "
        f"| input={n_input} "
        f"| RSS {rss_before:.1f}->{rss_after:.1f} GB "
        f"| MPS driver {mps_driver_before:.1f}->{mps_driver_after:.1f} GB"
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
    global _model, _tokenizer, _model_id, _device, _kv_pool

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
        low_cpu_mem_usage=True,
    ).to(_device)
    _model.eval()

    param_count = sum(p.numel() for p in _model.parameters()) / 1e9
    logger.info(f"Model loaded. Parameters: {param_count:.1f}B")

    # --- Configurable UT steps ---
    ut_steps_env = os.environ.get("OURO_UT_STEPS")
    if ut_steps_env is not None:
        ut_steps = int(ut_steps_env)
        old_steps = _model.config.total_ut_steps
        _model.config.total_ut_steps = ut_steps
        if hasattr(_model, "model") and hasattr(_model.model, "total_ut_steps"):
            _model.model.total_ut_steps = ut_steps
        logger.info(f"UT steps overridden: {old_steps} -> {ut_steps}")

    # --- Patch UniversalTransformerCache ---
    _apply_ut_cache_patches()

    # --- Initialize KV buffer pool ---
    num_layers = _model.config.num_hidden_layers
    total_ut_steps = _model.config.total_ut_steps
    num_kv_heads = _model.config.num_key_value_heads
    head_dim = getattr(
        _model.config, "head_dim",
        _model.config.hidden_size // _model.config.num_attention_heads,
    )
    max_slots = num_layers * total_ut_steps
    max_seq_len = int(os.environ.get("OURO_MAX_SEQ_LEN", "16384"))

    logger.info(
        f"Initializing KV pool: {num_layers} layers x {total_ut_steps} UT "
        f"steps = {max_slots} slots, max_seq={max_seq_len}"
    )

    _kv_pool = _KVBufferPool(
        max_slots=max_slots,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
        device=_device,
        num_layers=num_layers,
        total_ut_steps=total_ut_steps,
    )

    logger.info(
        f"MPS memory after init: "
        f"allocated={_mps_allocated_gb():.2f} GB, "
        f"driver={_mps_driver_gb():.2f} GB"
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
