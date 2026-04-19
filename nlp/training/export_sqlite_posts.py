#!/usr/bin/env python3
"""Export post content from ListenAI SQLite into JSONL for manual or LLM labeling."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/raw_posts.jsonl"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT id, content FROM posts WHERE length(trim(content)) > 0 ORDER BY id"
    ).fetchall()
    conn.close()

    random.seed(args.seed)
    if len(rows) > args.limit:
        rows = random.sample(rows, args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for pid, content in rows:
            rec = {"id": pid, "text": content.strip()}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
