#!/usr/bin/env python3
"""
Benchmark Ouro-2.6B on Apple Silicon via PyTorch MPS (Metal GPU).

Tests total_ut_steps=2/3/4 with both a simple prompt and a production-shaped
RealizationPacket prompt, reporting tok/s and wall-clock time.

Usage:
    python scripts/benchmark_ouro_mps.py                           # auto-detect local model
    python scripts/benchmark_ouro_mps.py --model-path /path/to/ouro
    python scripts/benchmark_ouro_mps.py --max-new-tokens 128
"""

import argparse
import json
import os
import sys
import time

import torch


def check_environment():
    import transformers

    print(f"  torch:        {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  MPS built:     {torch.backends.mps.is_built()}")

    major, minor, *_ = transformers.__version__.split(".")
    if int(major) >= 4 and int(minor) >= 56:
        print("  WARNING: transformers >= 4.56.0 may have Ouro compatibility issues.")
        print("           HuggingFace recommends transformers <= 4.54.1")

    if not torch.backends.mps.is_available():
        print("  WARNING: MPS not available, falling back to CPU. This will be slow.")
        return "cpu"
    return "mps"


SYSTEM_PROMPT = """\
You are a response realization engine.

You are given a structured response plan as JSON.

Rules:
1. Preserve all payload data exactly.
2. Do not invent facts, tools, or capabilities.
3. If uncertainty is specified, maintain it.
4. If clarification is required, include it clearly.
5. Combine adjacent units naturally into flowing prose.
6. Do not mention the plan or internal structure.
7. Output only the final candidate response text.
8. Match the specified tone.
9. Keep the response concise unless verbosity is set to high.
"""

PRODUCTION_PACKET = json.dumps({
    "mode": "plan_realization",
    "tone": "neutral",
    "verbosity": "medium",
    "analysis": {
        "text": "Hello! I'm Austin. It is nice to finally meet you my friend.",
        "intent": "greeting",
        "confidence": 0.92,
        "response_strategy": "can_respond",
        "speech_act": {
            "category": "expressive",
            "sub_type": "greeting",
            "confidence": 0.88,
            "is_question": False,
            "is_imperative": False,
        },
        "discourse": {
            "addressee": "bot",
            "confidence": 0.95,
            "direct_address_detected": True,
        },
        "sentiment": {"label": "positive", "confidence": 0.85},
        "entities": [
            {"text": "Austin", "type": "person", "confidence": 0.9}
        ],
        "slots": None,
        "epistemic_status": "known",
        "related_beliefs": [],
        "events": [],
        "fact_verification": None,
    },
    "context": None,
    "plan": [
        {
            "type": "greeting",
            "variant": "reciprocal",
            "content": {
                "greeting_text": "Hello! Nice to meet you too, Austin!",
                "user_name": "Austin",
                "familiarity": "new_acquaintance",
            },
            "confidence": 0.92,
        },
        {
            "type": "self_introduction",
            "variant": "brief",
            "content": {
                "text": "I'm happy to chat with you.",
                "capability_hint": "I can help with questions, conversation, and more.",
            },
        },
    ],
})

SIMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Respond concisely."},
    {"role": "user", "content": "Hello, how are you today?"},
]

PRODUCTION_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": PRODUCTION_PACKET},
]


def find_model_path():
    candidates = [
        os.path.join("apps", "brain", "priv", "ml_models", "ouro"),
        os.path.join(os.path.dirname(__file__), "..", "apps", "brain", "priv", "ml_models", "ouro"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c
    return None


def load_model(model_path, device, ut_steps):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading model from {model_path}")
    print(f"  total_ut_steps={ut_steps}, device={device}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.total_ut_steps = ut_steps

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {param_count / 1e9:.2f}B parameters, dtype={dtype}")

    return model, tokenizer


def run_generation(model, tokenizer, messages, device, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    torch.mps.synchronize() if device == "mps" else None

    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    torch.mps.synchronize() if device == "mps" else None
    elapsed = time.perf_counter() - t0

    output_ids = outputs[0][input_len:]
    gen_tokens = len(output_ids)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    word_count = len(text.split())

    return {
        "elapsed_s": round(elapsed, 2),
        "gen_tokens": gen_tokens,
        "input_tokens": input_len,
        "tok_per_sec": round(tok_per_sec, 1),
        "word_count": word_count,
        "char_count": len(text),
        "text_preview": text[:120].replace("\n", " "),
    }


def run_benchmark(model_path, device, max_new_tokens, ut_steps_list):
    print("=" * 72)
    print(f"  Ouro-2.6B MPS Benchmark")
    print("=" * 72)
    print(f"  Device:          {device}")
    print(f"  max_new_tokens:  {max_new_tokens}")
    print(f"  UT steps to test: {ut_steps_list}")
    print()

    results = {}

    for ut_steps in ut_steps_list:
        print("-" * 72)
        print(f"  total_ut_steps = {ut_steps}")
        print("-" * 72)

        model, tokenizer = load_model(model_path, device, ut_steps)

        # Warmup: single short generation to trigger any JIT/compilation
        print("  Warming up...")
        warmup_msgs = [{"role": "user", "content": "Hi"}]
        _ = run_generation(model, tokenizer, warmup_msgs, device, max_new_tokens=5)
        print("  Warmup complete.")

        # Simple prompt
        print(f"\n  Simple prompt ({SIMPLE_MESSAGES[1]['content'][:40]}...):")
        r_simple = run_generation(model, tokenizer, SIMPLE_MESSAGES, device, max_new_tokens)
        print(f"    {r_simple['tok_per_sec']} tok/s | {r_simple['elapsed_s']}s | "
              f"{r_simple['gen_tokens']} tokens ({r_simple['word_count']} words)")
        print(f"    Input: {r_simple['input_tokens']} tokens")
        print(f"    Output: \"{r_simple['text_preview']}...\"")

        # Production prompt
        print(f"\n  Production RealizationPacket prompt:")
        r_prod = run_generation(model, tokenizer, PRODUCTION_MESSAGES, device, max_new_tokens)
        print(f"    {r_prod['tok_per_sec']} tok/s | {r_prod['elapsed_s']}s | "
              f"{r_prod['gen_tokens']} tokens ({r_prod['word_count']} words)")
        print(f"    Input: {r_prod['input_tokens']} tokens")
        print(f"    Output: \"{r_prod['text_preview']}...\"")

        results[ut_steps] = {"simple": r_simple, "production": r_prod}

        del model
        torch.mps.empty_cache() if device == "mps" else None

        print()

    # Summary table
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    print()
    header = (
        f"  {'UT Steps':<10} {'Prompt':<14} {'tok/s':>8} {'Time(s)':>9} "
        f"{'Gen Tok':>9} {'In Tok':>8} {'Words':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ut_steps in ut_steps_list:
        for label, key in [("Simple", "simple"), ("Production", "production")]:
            r = results[ut_steps][key]
            print(
                f"  {ut_steps:<10} {label:<14} {r['tok_per_sec']:>8} {r['elapsed_s']:>9} "
                f"{r['gen_tokens']:>9} {r['input_tokens']:>8} {r['word_count']:>7}"
            )
    print()
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ouro-2.6B on MPS")
    parser.add_argument("--model-path", type=str, default=None, help="Path to local model directory")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ut-steps", type=str, default="2,3,4", help="Comma-separated UT step counts")
    args = parser.parse_args()

    print()
    print("  Environment check:")
    device = check_environment()
    print()

    model_path = args.model_path or find_model_path()
    if model_path is None or not os.path.isdir(model_path):
        print(f"ERROR: Model directory not found. Tried auto-detection and --model-path={args.model_path}")
        print("       Run `mix ouro.download --model 2.6b` first, or pass --model-path")
        sys.exit(1)

    print(f"  Model path: {model_path}")

    ut_steps_list = [int(x.strip()) for x in args.ut_steps.split(",")]

    run_benchmark(model_path, device, args.max_new_tokens, ut_steps_list)


if __name__ == "__main__":
    main()
