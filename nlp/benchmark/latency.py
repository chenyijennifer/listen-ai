#!/usr/bin/env python3
"""Micro-benchmark: lexicon vs sklearn batch latency (p50/p95)."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lexicon import classify_text_lexicon
from ml_backend import SklearnSentimentModel


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def bench_lexicon(texts: list[str], rounds: int) -> list[float]:
    times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for tx in texts:
            classify_text_lexicon(tx)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_sklearn(model: SklearnSentimentModel, texts: list[str], rounds: int) -> list[float]:
    times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        model.predict_batch(texts)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts-file", type=Path, help="JSONL with 'text' field; else use built-in")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--model-dir", type=Path, default=_ROOT / "model_artifacts")
    args = parser.parse_args()

    if args.texts_file and args.texts_file.is_file():
        import json

        texts = []
        for line in args.texts_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])
    else:
        texts = [
            "Great update, love it.",
            "很糟的體驗，不推薦。",
            "普通使用紀錄。",
        ] * (args.batch // 3 + 1)
    texts = texts[: args.batch]

    lex_times = sorted(bench_lexicon(texts, args.rounds))
    print(f"Lexicon batch={len(texts)} rounds={args.rounds} (ms per batch)")
    print(f"  p50={percentile(lex_times, 50):.3f}  p95={percentile(lex_times, 95):.3f}  mean={statistics.mean(lex_times):.3f}")

    model = SklearnSentimentModel.try_load(args.model_dir)
    if model:
        ml_times = sorted(bench_sklearn(model, texts, args.rounds))
        print(f"Sklearn batch={len(texts)} rounds={args.rounds} (ms per batch)")
        print(f"  p50={percentile(ml_times, 50):.3f}  p95={percentile(ml_times, 95):.3f}  mean={statistics.mean(ml_times):.3f}")
    else:
        print("Sklearn artifacts missing; only lexicon timed.")

    print("\nNotes: run inside Docker for deploy-like numbers; compare docker image size with `docker image ls`.")
    print(f"NLP_SENTIMENT_BACKEND={os.getenv('NLP_SENTIMENT_BACKEND', '(auto)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
