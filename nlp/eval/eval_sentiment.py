#!/usr/bin/env python3
"""Evaluate lexicon baseline and sklearn model on the test split of labeled JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Allow running from repo root or nlp/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lexicon import classify_text_lexicon
from ml_backend import SklearnSentimentModel

LABELS = ["negative", "neutral", "positive"]


def load_test(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("split", "train") != "test":
            continue
        texts.append(row["text"])
        labels.append(row["label"].lower())
    return texts, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=_ROOT / "data" / "labeled_seed.jsonl")
    parser.add_argument("--model-dir", type=Path, default=_ROOT / "model_artifacts")
    args = parser.parse_args()

    texts, y_true = load_test(args.data)
    if not texts:
        raise SystemExit("No test rows (split=test) in dataset")

    y_lex = [classify_text_lexicon(t)[0] for t in texts]
    print("=== Lexicon (test) ===")
    print(f"accuracy: {accuracy_score(y_true, y_lex):.4f}")
    print(f"macro F1: {f1_score(y_true, y_lex, average='macro'):.4f}")
    print(f"weighted F1: {f1_score(y_true, y_lex, average='weighted'):.4f}")
    print(classification_report(y_true, y_lex, digits=4))
    print("confusion_matrix (rows=true, cols=pred)")
    print(confusion_matrix(y_true, y_lex, labels=LABELS))

    model = SklearnSentimentModel.try_load(args.model_dir)
    if not model:
        print("\n(sklearn artifacts missing; skip ML metrics)")
        return 0

    y_ml = [model.predict_batch([t])[0][0] for t in texts]
    print("\n=== TF-IDF + LogisticRegression (sklearn, test) ===")
    print(f"accuracy: {accuracy_score(y_true, y_ml):.4f}")
    print(f"macro F1: {f1_score(y_true, y_ml, average='macro'):.4f}")
    print(f"weighted F1: {f1_score(y_true, y_ml, average='weighted'):.4f}")
    print(classification_report(y_true, y_ml, digits=4))
    print("confusion_matrix (rows=true, cols=pred)")
    print(confusion_matrix(y_true, y_ml, labels=LABELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
