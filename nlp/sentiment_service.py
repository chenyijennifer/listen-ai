"""Route sentiment requests to lexicon or sklearn backend."""

from __future__ import annotations

import os
from pathlib import Path

from lexicon import classify_text_lexicon
from ml_backend import SklearnSentimentModel

_backend_name: str | None = None
_model: SklearnSentimentModel | None = None


def _default_model_dir() -> Path:
    env = os.getenv("NLP_MODEL_DIR", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "model_artifacts"


def _resolve_backend() -> tuple[str, SklearnSentimentModel | None]:
    global _backend_name, _model
    if _backend_name is not None:
        return _backend_name, _model

    explicit = os.getenv("NLP_SENTIMENT_BACKEND", "").strip().lower()
    model = SklearnSentimentModel.try_load(_default_model_dir())

    if explicit == "lexicon":
        _backend_name, _model = "lexicon", None
    elif explicit == "sklearn":
        _backend_name = "sklearn" if model else "lexicon"
        _model = model if model else None
    else:
        # auto: prefer sklearn when artifacts exist
        if model:
            _backend_name, _model = "sklearn", model
        else:
            _backend_name, _model = "lexicon", None

    return _backend_name, _model


def reset_backend_cache_for_tests() -> None:
    global _backend_name, _model
    _backend_name, _model = None, None


def active_backend() -> str:
    name, _ = _resolve_backend()
    return name


def classify_batch(texts: list[str]) -> list[tuple[str, int]]:
    name, model = _resolve_backend()
    if name == "sklearn" and model is not None:
        return model.predict_batch(texts)
    return [classify_text_lexicon(t) for t in texts]
