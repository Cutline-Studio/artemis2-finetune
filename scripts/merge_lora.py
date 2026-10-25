"""Merge a LoRA adapter into the base model and export a standalone folder.

On an H100 this needs enough disk for base + merged (~60–80 GB for 27B bf16).
Does not upload anywhere.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(json.dumps({"event": "merge_start", "base": args.base, "adapter": args.adapter}))
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(json.dumps({"event": "merge_done", "out": str(out)}))


if __name__ == "__main__":
    main()
