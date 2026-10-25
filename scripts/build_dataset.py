"""Assemble train/eval JSONL from hand + generated examples. Dedupes and validates."""
from __future__ import annotations

import argparse, json, random, hashlib
from collections import Counter
from pathlib import Path

from hand_examples import all_hand
from persona import system_prompt, load_facts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def validate(ex: dict) -> str | None:
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return "bad messages"
    roles = [m.get("role") for m in msgs]
    if roles[0] != "system":
        return "no system"
    if "user" not in roles or "assistant" not in roles:
        return "missing user/assistant"
    for m in msgs:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            return f"bad role {m.get('role')}"
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return "empty content"
    # last must be assistant
    if msgs[-1]["role"] != "assistant":
        return "last not assistant"
    return None


def content_key(ex: dict) -> str:
    # dedupe on user turns + final assistant (ignore system, which is shared)
    parts = [(m["role"], m["content"]) for m in ex["messages"] if m["role"] != "system"]
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def load_generated(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def strip_meta(ex: dict) -> dict:
    return {
        "messages": ex["messages"],
        "category": ex.get("category", "unknown"),
        "source": ex.get("source", "unknown"),
        "license": ex.get("license", "unknown"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=Path, default=DATA / "raw" / "generated.jsonl")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--eval-frac", type=float, default=0.08)
    ap.add_argument("--eval-min-per-cat", type=int, default=4)
    args = ap.parse_args()

    # ensure system prompt is current on hand examples
    facts = load_facts()
    sys = system_prompt(facts)

    examples = []
    for ex in all_hand():
        ex = dict(ex)
        ex["messages"] = [{"role": "system", "content": sys}] + [
            m for m in ex["messages"] if m["role"] != "system"
        ]
        examples.append(ex)
    examples.extend(load_generated(args.generated))

    clean, skipped = [], Counter()
    seen = set()
    for ex in examples:
        err = validate(ex)
        if err:
            skipped[err] += 1
            continue
        k = content_key(ex)
        if k in seen:
            skipped["dup"] += 1
            continue
        # drop teacher leaks
        asst = " ".join(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        leak = ("I am Nemotron", "I'm Nemotron", "I am Qwen", "I'm an NVIDIA", "As Nemotron")
        if any(x.lower() in asst.lower() for x in leak):
            skipped["teacher_leak"] += 1
            continue
        seen.add(k)
        clean.append(strip_meta(ex))

    rng = random.Random(args.seed)
    # stratified-ish split by category
    by_cat: dict[str, list] = {}
    for ex in clean:
        by_cat.setdefault(ex["category"], []).append(ex)
    train, eval_ = [], []
    for cat, xs in by_cat.items():
        rng.shuffle(xs)
        n_eval = max(args.eval_min_per_cat, int(round(len(xs) * args.eval_frac)))
        n_eval = min(n_eval, max(1, len(xs) // 5))  # never take more than 20%
        if len(xs) < args.eval_min_per_cat * 2:
            n_eval = max(1, len(xs) // 5)
        eval_.extend(xs[:n_eval])
        train.extend(xs[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_)

    DATA.mkdir(parents=True, exist_ok=True)
    train_path = DATA / "train.jsonl"
    eval_path = DATA / "eval.jsonl"
    with train_path.open("w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with eval_path.open("w") as f:
        for ex in eval_:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "train": len(train),
        "eval": len(eval_),
        "total": len(clean),
        "skipped": dict(skipped),
        "train_by_category": dict(Counter(e["category"] for e in train)),
        "eval_by_category": dict(Counter(e["category"] for e in eval_)),
        "sources": dict(Counter(e["source"] for e in clean)),
        "licenses": dict(Counter(e["license"] for e in clean)),
        "seed": args.seed,
        "system_prompt_sha1": hashlib.sha1(sys.encode()).hexdigest()[:12],
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
