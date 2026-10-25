# Vast.ai RTX A6000 run — Qwen/Qwen3.5-4B bf16 LoRA (2026-10-25) — ABORTED, no results

- Offer 21004756: 1x RTX A6000 48 GB, Kansas US, $0.40/hr GPU (+$0.017/hr storage, 60 GB disk), reliability 0.991, verified.
- Instance 57720677, image `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`, created 4:52 PM ET.
- Config: `configs/a6000_small.yaml` (same base model as the T4 run, bf16, no 4-bit, r=32, 2 epochs, seq 2048).
- Script: `vastai/run_a6000.sh` (pip install pinned stack, train, eval_prompts.py on 37 eval rows).

## What happened
- Setup worked: torch 2.14.1+cu126, transformers 5.20.2, trl 1.16.0, peft 0.21.2; setup 73 s.
- Training ran at ~7 s/step (96 steps total, ~12 min estimated), ~10.7 GB VRAM.
- The container was stopped from outside twice (instance intended_status -> "stopped"), once at ~4:53 PM ET
  and again during training (after step 14/96). The restart then failed with `resources_unavailable`, and the
  instance disappeared from the account (API: no_such_instance).
- At the same time the account's credit went to $0: an extra $10 card top-up at 4:55 PM ET and a $19.64
  credit transfer to another Vast account (764133) at 5:09 PM ET, plus other instances (57721079-57721601)
  that this run did not create. Run stopped rather than add credit.
- No adapter, loss or probe outputs were recovered. Charges for this instance: $0.109 GPU + $0.006 storage.
