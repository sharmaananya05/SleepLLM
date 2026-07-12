"""
models/load_model.py
=====================
A memory-safe sanity-check script: loads a SMALL pretrained model and runs
one generation, to prove your PyTorch + CUDA + transformers stack works
end-to-end before we build the custom SleepLLM backbone on top of it.

This is NOT part of the SleepLLM architecture itself -- it's an
environment smoke test (like "Hello, World!" for your GPU setup). The real
SleepLLM backbone (models/backbone.py, coming in Module 3) is a small
custom Transformer we train from scratch, sized to fit your 4GB GPU via
config/debug_laptop.yaml.

Why the earlier version crashed, and what's fixed here:
--------------------------------------------------------
1. `low_cpu_mem_usage=True` -- streams weights from disk directly into
   their final dtype, instead of materializing a full fp32 copy in CPU
   RAM first (the actual cause of your `memory allocation ... failed`
   error).
2. `dtype=` (not the deprecated `torch_dtype=`) set to fp16 on GPU.
3. `device_map="auto"` places weights directly on GPU while reading them
   off disk, instead of "load fully to CPU, then .to(device)" -- which
   would transiently need 2x the memory during the copy.
4. MODEL_NAME defaults to "distilgpt2" (~330MB) -- small enough to load
   comfortably on an 8GB RAM / 4GB VRAM laptop. If you need to test a
   bigger model, do it on Lightning AI, not here.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"  # ~330MB, safe for 8GB RAM / 4GB VRAM.
# Swap to "gpt2" (~550MB) if you want the slightly larger baseline GPT-2.
# Do NOT put a 1B+ parameter model here -- test those on Lightning AI.


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 60)
    print(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("=" * 60)
    print(f"Loading model '{MODEL_NAME}'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()  # disables dropout -- we're doing inference, not training

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params / 1e6:.1f}M")

    print("=" * 60)
    print("Running a generation smoke test...")
    prompt = "The Sleep paradigm for language models works by"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():  # no gradients needed for inference -> saves memory
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,  # greedy decoding: deterministic, reproducible
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("-" * 60)
    print(generated_text)
    print("-" * 60)
    print("Environment smoke test PASSED.")


if __name__ == "__main__":
    main()
