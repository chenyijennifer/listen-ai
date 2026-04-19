"""TF-IDF + LogisticRegression inference (loads joblib artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

LABEL_ORDER = ("negative", "neutral", "positive")


class SklearnSentimentModel:
    def __init__(self, model_dir: Path) -> None:
        model_dir = model_dir.resolve()
        self._vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self._clf = joblib.load(model_dir / "model.joblib")
        meta_path = model_dir / "labels.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._classes: list[str] = list(meta["classes"])
        else:
            self._classes = list(getattr(self._clf, "classes_", LABEL_ORDER))

        self._idx = {c: i for i, c in enumerate(self._classes)}

    @classmethod
    def try_load(cls, model_dir: Path | None) -> SklearnSentimentModel | None:
        if not model_dir:
            return None
        p = model_dir.resolve()
        if not (p / "vectorizer.joblib").is_file() or not (p / "model.joblib").is_file():
            return None
        return cls(p)

    def predict_batch(self, texts: list[str]) -> list[tuple[str, int]]:
        if not texts:
            return []
        X = self._vectorizer.transform(texts)
        labels = self._clf.predict(X)
        probas = self._clf.predict_proba(X)
        out: list[tuple[str, int]] = []
        i_neg = self._idx.get("negative", 0)
        i_pos = self._idx.get("positive", len(self._classes) - 1)
        for lab, row in zip(labels, probas, strict=True):
            p_pos = float(row[i_pos])
            p_neg = float(row[i_neg])
            # Intensity: positive vs negative margin, scaled for dashboard display
            margin = p_pos - p_neg
            score = int(np.clip(round(margin * 100), -100, 100))
            out.append((str(lab), score))
        return out
