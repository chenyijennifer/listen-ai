#!/usr/bin/env python3
"""Assign stratified train/val/test splits to labeled JSONL (text + label, no split yet)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must sum to 1")

    by_label: dict[str, list[dict]] = defaultdict(list)
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "split" in row:
            del row["split"]
        by_label[row["label"]].append(row)

    rnd = random.Random(args.seed)
    out_rows: list[dict] = []
    for label, items in sorted(by_label.items()):
        rnd.shuffle(items)
        n = len(items)
        n_train = int(round(n * args.train))
        n_val = int(round(n * args.val))
        n_test = max(0, n - n_train - n_val)
        n_train = n - n_val - n_test
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        assert len(splits) == n, (label, n, len(splits))
        for row, sp in zip(items, splits, strict=True):
            row2 = {**row, "split": sp}
            out_rows.append(row2)

    rnd.shuffle(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
