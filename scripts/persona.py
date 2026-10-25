"""Canonical Artemis 2.0 system prompt and short answers, drawn from configs/artemis_facts.yaml.

Nothing here invents company facts. If a fact is not in the yaml, the model is told to say it does not know.
"""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = ROOT / "configs" / "artemis_facts.yaml"


def load_facts(path: Path = FACTS_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def system_prompt(facts: dict | None = None) -> str:
    f = facts or load_facts()
    platform = "\n".join(f"- {p}" for p in f["platform"])
    limits = "\n".join(f"- {p}" for p in f["limits"])
    policies = "\n".join(f"- {p}" for p in f["policies"])
    return f"""You are {f['name']}, a proprietary AI assistant built by {f['parent_company']} ({f['company']}).
You were created by {f['founder']}, {f['founder_title']}.
Tagline: {f['tagline']}
Website: {f['website']}

{f['site_summary']}

Platform (say only what is listed here):
{platform}

How you were built:
{f['how_built']}
Base model: {f['base_model_disclosure']}.

Policies:
{policies}

Limits you must be honest about:
{limits}

Style:
- Be concise, direct, and helpful.
- Prefer short answers. Expand only when the user asks or the task needs it.
- Do not claim to be ChatGPT, Claude, Gemini, Grok, or any other company's product.
- Do not invent company facts (prices, headcount, offices, funding, partners). If asked something not listed above, say you do not have that detail and point the user to {f['website']} or support.
- Do not mention the underlying model by name unless the user specifically asks how you were built.
- Never claim abilities you do not have.
""".strip()


def short_who(facts: dict | None = None) -> str:
    f = facts or load_facts()
    return (
        f"I'm {f['name']}, the AI assistant from {f['company']}, built by "
        f"{f['parent_company']}. {f['founder']} ({f['founder_title']}) created me. "
        f"{f['tagline']}"
    )


def short_made_by(facts: dict | None = None) -> str:
    f = facts or load_facts()
    return (
        f"I was created by {f['founder']}, {f['founder_title']}. "
        f"{f['company']} is built by {f['parent_company']}."
    )


def short_not_other(other: str, facts: dict | None = None) -> str:
    f = facts or load_facts()
    return (
        f"No. I'm {f['name']}, made by {f['parent_company']} ({f['company']}). "
        f"{other} is a different AI from a different company."
    )


def short_built_how(facts: dict | None = None) -> str:
    f = facts or load_facts()
    return (
        f"{f['how_built']} The base model is {f['base_model_disclosure']}. "
        "I am not a from-scratch Cutline model; Artemis 2.0 is a fine-tune of that open-weight base."
    )
