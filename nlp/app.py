import os
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel

from lexicon import classify_text_lexicon
from sentiment_service import active_backend, classify_batch

app = FastAPI(title="listen-ai-nlp")


def classify_text(text: str) -> tuple[str, int]:
    """Backward-compatible name for lexicon-only classification (eval / tests)."""
    return classify_text_lexicon(text)


class SentimentRequest(BaseModel):
    texts: list[str]


class SentimentItem(BaseModel):
    text: str
    label: str
    score: int


class SentimentResponse(BaseModel):
    sentiment_percentage: dict[str, float]
    classifications: list[SentimentItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "nlp",
        "port": os.getenv("NLP_PORT", "8001"),
        "sentiment_backend": active_backend(),
    }


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest) -> SentimentResponse:
    results: list[SentimentItem] = []
    counts = Counter({"positive": 0, "neutral": 0, "negative": 0})

    batch = classify_batch(req.texts)
    for text, (label, score) in zip(req.texts, batch, strict=True):
        counts[label] += 1
        results.append(SentimentItem(text=text, label=label, score=score))

    total = max(1, len(req.texts))
    sentiment_percentage = {
        "positive": round((counts["positive"] / total) * 100, 2),
        "neutral": round((counts["neutral"] / total) * 100, 2),
        "negative": round((counts["negative"] / total) * 100, 2),
    }

    return SentimentResponse(
        sentiment_percentage=sentiment_percentage,
        classifications=results,
    )
