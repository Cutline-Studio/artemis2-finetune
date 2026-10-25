# Artemis 2.0 — fine-tune on one Brev H100 80GB

**Do not run this until Latoya explicitly approves creating the GPU.**
Nothing in this repo creates cloud resources by itself.

## What this trains

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0; verified on Hugging Face).
- Method: QLoRA (4-bit NF4) + LoRA r=64 on attention/MLP projections.
- Data: `data/train.jsonl` / `data/eval.jsonl` (Artemis persona SFT).
- Target hardware: 1× H100 80GB. Cheapest Brev option observed: Hyperstack @ **$3.00/hr** (estimate subject to change).

## Time and cost estimate (labelled estimate)

Assumptions: ~800–1,200 train rows, seq len 4096, 2 epochs, batch 1 × grad accum 8, QLoRA 4-bit on H100 80GB.

| Step | Estimate |
|------|----------|
| Pull base weights (~55 GB) | 10–20 min |
| Train 2 epochs | 2.5–5 hr |
| Eval + merge export | 20–40 min |
| **Total GPU time** | **≈ 3.5–6.5 hr** |
| **Cost at $3.00/hr** | **≈ $12–$24** (estimate) |

These are estimates, not quotes. Actual time depends on dataset size after generation, tokens/example, and Brev pull speed.

## Create the instance (only after approval)

```bash
# From a machine logged into Latoya's Brev org (support@artemisai-bycutlineindustries.studio)
brev login
brev create artemis2-h100 --gpu h100 --n 1 --cloud hyperstack   # confirm flags with `brev create -h`
brev shell artemis2-h100
```

If Hyperstack H100 is unavailable, MassedCompute H100 was next at about **$3.28/hr** (estimate).

## On the H100

```bash
git clone git@github.com:lpittman-blip/artemis2-finetune.git
cd artemis2-finetune
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Optional: Hugging Face token for higher rate limits (public model; token not required)
# export HF_TOKEN=...

python scripts/train_sft.py --config configs/train_qlora_h100.yaml

python scripts/eval_prompts.py \
  --base Qwen/Qwen3.8-27B \
  --adapter outputs/artemis2-qlora/adapter \
  --load-4bit

# Optional full merge (needs ~80GB free disk for bf16 export)
python scripts/merge_lora.py \
  --base Qwen/Qwen3.8-27B \
  --adapter outputs/artemis2-qlora/adapter \
  --out outputs/artemis2-merged-bf16
```

Copy adapters / metrics off the box (object storage or `brev scp`) before stopping.

## Stop billing

```bash
exit   # leave the shell
brev stop artemis2-h100
# or: brev delete artemis2-h100   # only if you are sure the outputs are saved
```

Leaving an H100 running overnight at $3/hr is about $72 (estimate). Always stop when finished.

## CPU smoke test (already run on the shared box)

```bash
cd /workspace/artemis2-finetune
source .venv/bin/activate
PYTHONPATH=scripts python scripts/build_dataset.py
PYTHONPATH=scripts python scripts/train_sft.py --config configs/train_smoke_cpu.yaml
PYTHONPATH=scripts python scripts/eval_prompts.py \
  --base Qwen/Qwen3.5-0.8B \
  --adapter outputs/smoke-cpu/adapter \
  --cpu --max-samples 8
```

See `smoke/SMOKE_REPORT.md` for the recorded command, versions, seed, and raw output.
