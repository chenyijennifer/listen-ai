from __future__ import annotations

import os
import unittest
from pathlib import Path

from lexicon import classify_text_lexicon
from ml_backend import SklearnSentimentModel
from sentiment_service import classify_batch, reset_backend_cache_for_tests


class LexiconTests(unittest.TestCase):
    def test_positive_and_negation(self) -> None:
        lab, score = classify_text_lexicon("not bad at all")
        self.assertEqual(lab, "positive")
        self.assertGreater(score, 0)

    def test_chinese_positive(self) -> None:
        lab, _ = classify_text_lexicon("這次更新很好")
        self.assertEqual(lab, "positive")

    def test_neutral(self) -> None:
        lab, score = classify_text_lexicon("版本號更新為 2.4.1。")
        self.assertEqual(lab, "neutral")
        self.assertEqual(score, 0)


class ServiceRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_backend_cache_for_tests()
        for key in ("NLP_SENTIMENT_BACKEND", "NLP_MODEL_DIR"):
            os.environ.pop(key, None)

    def test_lexicon_backend(self) -> None:
        os.environ["NLP_SENTIMENT_BACKEND"] = "lexicon"
        labels = [classify_batch(["great", "很糟"])[i][0] for i in range(2)]
        self.assertEqual(labels, ["positive", "negative"])

    def test_sklearn_when_artifacts_present(self) -> None:
        model_dir = Path(__file__).resolve().parent.parent / "model_artifacts"
        if not (model_dir / "model.joblib").is_file():
            self.skipTest("model artifacts not built")
        os.environ["NLP_SENTIMENT_BACKEND"] = "sklearn"
        os.environ["NLP_MODEL_DIR"] = str(model_dir)
        out = classify_batch(["體驗很糟，不推薦購買。", "整體很滿意，推薦大家試試。"])
        self.assertEqual(out[0][0], "negative")
        self.assertEqual(out[1][0], "positive")


class SklearnModelTests(unittest.TestCase):
    def test_predict_batch_empty(self) -> None:
        model_dir = Path(__file__).resolve().parent.parent / "model_artifacts"
        model = SklearnSentimentModel.try_load(model_dir)
        if model is None:
            self.skipTest("no model")
        self.assertEqual(model.predict_batch([]), [])


if __name__ == "__main__":
    unittest.main()
