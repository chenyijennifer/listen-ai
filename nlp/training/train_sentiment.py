#!/usr/bin/env python3
"""Train TF-IDF + LogisticRegression on labeled JSONL and write model_artifacts/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

LABELS = ("negative", "neutral", "positive")


def load_rows(path: Path) -> tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    splits: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        lab = row["label"].strip().lower()
        if lab not in LABELS:
            raise ValueError(f"Bad label {lab!r} in {path}")
        texts.append(row["text"])
        labels.append(lab)
        splits.append(str(row.get("split", "train")).strip().lower())
    return texts, labels, splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "labeled_seed.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "model_artifacts")
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument(
        "--fit-splits",
        type=str,
        default="train,val",
        help="Comma-separated splits used for fitting (test is always held out for your own eval).",
    )
    args = parser.parse_args()

    fit_set = {x.strip() for x in args.fit_splits.split(",") if x.strip()}
    texts, labels, splits = load_rows(args.data)
    train_x = [t for t, s in zip(texts, splits, strict=True) if s in fit_set]
    train_y = [y for y, s in zip(labels, splits, strict=True) if s in fit_set]
    val_x = [t for t, s in zip(texts, splits, strict=True) if s == "val"]
    val_y = [y for y, s in zip(labels, splits, strict=True) if s == "val"]

    if len(train_x) < 10:
        raise SystemExit("Need at least ~10 rows in --fit-splits")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(1, 4),
                    min_df=1,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    C=1.0,
                    solver="saga",
                ),
            ),
        ]
    )

    pipeline.fit(train_x, train_y)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["tfidf"], args.out / "vectorizer.joblib")
    joblib.dump(pipeline.named_steps["clf"], args.out / "model.joblib")
    (args.out / "labels.json").write_text(
        json.dumps({"classes": list(pipeline.named_steps["clf"].classes_)}),
        encoding="utf-8",
    )

    def report_split(name: str, xs: list[str], ys: list[str]) -> None:
        if not xs:
            return
        pred = pipeline.predict(xs)
        print(f"\n=== {name} ({len(xs)} samples) ===")
        print(classification_report(ys, pred, digits=4))
        print("confusion_matrix (rows=true, cols=pred)")
        print(confusion_matrix(ys, pred, labels=list(LABELS)))

    report_split("fit", train_x, train_y)
    if "val" not in fit_set:
        report_split("val (held out from fit)", val_x, val_y)
    test_x = [t for t, s in zip(texts, splits, strict=True) if s == "test"]
    test_y = [y for y, s in zip(labels, splits, strict=True) if s == "test"]
    report_split("test (always held out)", test_x, test_y)
    print(f"\nWrote artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
