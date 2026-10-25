# artemis2-finetune

Prepare-only repo for **Artemis 2.0**: a LoRA/QLoRA supervised fine-tune of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) so the
model answers as Artemis (Cutline Industries / Artemis AI).

**No cloud GPUs are created from this repo.** Training on Brev happens only after
explicit approval. See [`RUN.md`](RUN.md).

## Layout

```
configs/artemis_facts.yaml     # verified product facts (no inventions)
configs/train_qlora_h100.yaml  # H100 QLoRA config
configs/train_smoke_cpu.yaml   # CPU smoke config (Qwen3.5-0.8B)
data/train.jsonl               # SFT train split
data/eval.jsonl                # held-out eval
data/DATA_CARD.md              # counts, sources, licenses, seeds
scripts/                       # build, generate, train, eval, merge
smoke/                         # recorded CPU smoke output
```

## Model choice

`Qwen/Qwen3.8-27B` exists on Hugging Face (Apache-2.0, ~27B dense, `qwen3_5`
architecture). It is a native vision-language checkpoint; text chat works through
the standard Transformers chat template with `enable_thinking=false` for SFT.

## Persona source

System prompt and identity answers are derived from:

- `Cutline-Industries/gateway/configs/business.yaml` (founder, company)
- `cutline-site/index.html` (tagline, platform surfaces)
- Existing runtime prompts in `artemis-serve.py` / `ChatBox.tsx` / `artemis-cli`

Unknown commercial details (prices, headcount, funding) are refused, not invented.
