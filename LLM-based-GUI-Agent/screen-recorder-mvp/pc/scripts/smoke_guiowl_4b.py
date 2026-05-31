"""
Smoke test: load mPLUG/GUI-Owl-1.5-4B-Instruct with Transformers (Mobile-Agent-v3.5 Quick Start path).
Requires: venv .venv-guiowl-hf + requirements-guiowl-4b.txt
"""
from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main() -> int:
    p = argparse.ArgumentParser(description="GUI-Owl 1.5 4B smoke test (Transformers)")
    p.add_argument(
        "--model",
        default="mPLUG/GUI-Owl-1.5-4B-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--gpu-mem",
        default="6GiB",
        help='Per accelerate max_memory for cuda:0, e.g. "6GiB" on 8GB laptop GPU',
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; install NVIDIA driver + PyTorch with CUDA.", file=sys.stderr)
        return 1

    dtype = torch.float16
    max_memory = {0: args.gpu_mem, "cpu": "24GiB"}

    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading model (first run downloads weights from Hugging Face; large)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()

    # 1x1 占位图，仅验证多模态前向能跑通
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "In one short sentence, describe the main color of this image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print("Generating...", flush=True)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    trimmed = [
        out[len(inp) :]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("OK:", text[0] if text else "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
