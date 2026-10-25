"""Concatenate raw generation JSONL shards into data/raw/generated.jsonl."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
shards = [
    RAW / "generated_partial.jsonl",
    RAW / "gen_coding.jsonl",
    RAW / "gen_general.jsonl",
    RAW / "generated.jsonl",
]
out = RAW / "generated_all.jsonl"
n = 0
with out.open("w") as w:
    for p in shards:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                if line.strip():
                    w.write(line if line.endswith("\n") else line + "\n")
                    n += 1
print({"wrote": str(out), "lines": n})
