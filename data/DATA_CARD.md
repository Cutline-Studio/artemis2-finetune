# Artemis 2.0 SFT data card

Generated: 2026-10-25 20:17 UTC

## Summary

- Train rows: **377**
- Eval rows: **37**
- Total clean rows: **414**
- Skipped during build: `{'dup': 89}`
- System prompt sha1 (12): `2caa10d6129c`
- Build seed: `20261025`

### Train by category

- coding: 202
- general: 109
- identity: 33
- multiturn: 7
- safety: 26

### Eval by category

- coding: 18
- general: 10
- identity: 4
- multiturn: 1
- safety: 4

## Sources and licenses

| Source | License / terms | Count |
|---|---|---|
| `hand` | see licenses map | 85 |
| `nvidia-build:nvidia/nemotron-3-ultra-550b-a55b` | see licenses map | 329 |

License strings observed:

- `Cutline-Industries-internal` × 85
- `generated-for-Cutline-Industries; teacher-output; not redistributed as teacher weights` × 329

### Notes on licenses

- **hand / Cutline-Industries-internal**: authored for Cutline Industries from
  `configs/artemis_facts.yaml`, which itself is copied from the Cutline Industries
  gateway business config and the public Artemis site copy. Not third-party text.
- **nvidia-build:...**: assistant completions generated with the NVIDIA Build API
  teacher listed below. Outputs are produced for Cutline Industries' fine-tune;
  teacher model *weights* are not redistributed. Do not treat teacher outputs as
  a license to rehost the teacher.
- No third-party SFT corpora (UltraChat, OASST, etc.) were mixed into this release,
  so there is no additional dataset license beyond the above.

## Generation

- Teacher model: `nvidia/nemotron-3-ultra-550b-a55b`
- Endpoint: `https://integrate.api.nvidia.com/v1/chat/completions`
- Script: `scripts/generate_sft.py`
- Seeds: `[20261025, 20261026]`
- Rate limit target: ~18–35 requests/minute (API budget ~40/min)
- Thinking disabled for Nemotron 3 via `chat_template_kwargs.enable_thinking=false`

## Format

Each JSONL line:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"category":"identity|coding|general|safety|multiturn","source":"...","license":"..."}
```

Multi-turn examples include additional user/assistant turns after the system message.

## Persona facts (no inventions)

Company facts come only from `configs/artemis_facts.yaml` (founder Latoya Pittman,
Cutline Industries / Artemis AI, tagline, platform surfaces, website). Plan prices,
headcount, funding, and office locations are deliberately omitted; the model is
trained to say it does not know rather than invent.

## Dedup / validation

`scripts/build_dataset.py` validates roles/content, drops teacher-name leaks,
and dedupes on (non-system turns) SHA1.

