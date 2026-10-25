"""Write data/DATA_CARD.md from data/summary.json and generation metadata."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    summary = json.loads((DATA / "summary.json").read_text())
    gen_meta = {
        "teacher_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "teacher_endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "generation_script": "scripts/generate_sft.py",
        "generation_seeds": [20261025, 20261026],
        "hand_script": "scripts/hand_examples.py",
        "facts_file": "configs/artemis_facts.yaml",
        "build_script": "scripts/build_dataset.py",
        "build_seed": summary.get("seed"),
    }
    lines = [
        "# Artemis 2.0 SFT data card",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Train rows: **{summary['train']}**",
        f"- Eval rows: **{summary['eval']}**",
        f"- Total clean rows: **{summary['total']}**",
        f"- Skipped during build: `{summary.get('skipped')}`",
        f"- System prompt sha1 (12): `{summary.get('system_prompt_sha1')}`",
        f"- Build seed: `{summary.get('seed')}`",
        "",
        "### Train by category",
        "",
    ]
    for k, v in sorted(summary.get("train_by_category", {}).items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "### Eval by category", ""]
    for k, v in sorted(summary.get("eval_by_category", {}).items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Sources and licenses", ""]
    lines.append("| Source | License / terms | Count |")
    lines.append("|---|---|---|")
    # combine source counts
    for src, n in sorted(summary.get("sources", {}).items()):
        lic = "see licenses map"
        lines.append(f"| `{src}` | {lic} | {n} |")
    lines += ["", "License strings observed:", ""]
    for lic, n in sorted(summary.get("licenses", {}).items()):
        lines.append(f"- `{lic}` × {n}")
    lines += [
        "",
        "### Notes on licenses",
        "",
        "- **hand / Cutline-Industries-internal**: authored for Cutline Industries from",
        "  `configs/artemis_facts.yaml`, which itself is copied from the Cutline Industries",
        "  gateway business config and the public Artemis site copy. Not third-party text.",
        "- **nvidia-build:...**: assistant completions generated with the NVIDIA Build API",
        "  teacher listed below. Outputs are produced for Cutline Industries' fine-tune;",
        "  teacher model *weights* are not redistributed. Do not treat teacher outputs as",
        "  a license to rehost the teacher.",
        "- No third-party SFT corpora (UltraChat, OASST, etc.) were mixed into this release,",
        "  so there is no additional dataset license beyond the above.",
        "",
        "## Generation",
        "",
        f"- Teacher model: `{gen_meta['teacher_model']}`",
        f"- Endpoint: `{gen_meta['teacher_endpoint']}`",
        f"- Script: `{gen_meta['generation_script']}`",
        f"- Seeds: `{gen_meta['generation_seeds']}`",
        f"- Rate limit target: ~18–35 requests/minute (API budget ~40/min)",
        f"- Thinking disabled for Nemotron 3 via `chat_template_kwargs.enable_thinking=false`",
        "",
        "## Format",
        "",
        "Each JSONL line:",
        "",
        "```json",
        '{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],'
        '"category":"identity|coding|general|safety|multiturn","source":"...","license":"..."}',
        "```",
        "",
        "Multi-turn examples include additional user/assistant turns after the system message.",
        "",
        "## Persona facts (no inventions)",
        "",
        "Company facts come only from `configs/artemis_facts.yaml` (founder Latoya Pittman,",
        "Cutline Industries / Artemis AI, tagline, platform surfaces, website). Plan prices,",
        "headcount, funding, and office locations are deliberately omitted; the model is",
        "trained to say it does not know rather than invent.",
        "",
        "## Dedup / validation",
        "",
        "`scripts/build_dataset.py` validates roles/content, drops teacher-name leaks,",
        "and dedupes on (non-system turns) SHA1.",
        "",
    ]
    (DATA / "DATA_CARD.md").write_text("\n".join(lines) + "\n")
    (DATA / "generation_meta.json").write_text(json.dumps(gen_meta, indent=2) + "\n")
    print("wrote", DATA / "DATA_CARD.md")


if __name__ == "__main__":
    main()
