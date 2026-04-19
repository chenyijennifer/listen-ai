# NLP Service

Default inference uses **TF--IDF + LogisticRegression** when `model_artifacts/` is present; otherwise it falls back to the lexicon baseline. Override with `NLP_SENTIMENT_BACKEND=lexicon` or `sklearn`, and `NLP_MODEL_DIR` if artifacts live outside the default path.

## Prerequisites

- Python 3.11+

## Run Without Docker

1. Open a terminal in this folder:

```bash
cd nlp
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Configure port:

```bash
export NLP_PORT=8001
```

5. Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port ${NLP_PORT:-8001}
```

## Health Check

```bash
curl http://localhost:8001/health
```

## Example Request

```bash
curl -X POST http://localhost:8001/sentiment \
  -H "Content-Type: application/json" \
  -d '{"texts":["great update","bad experience","這次更新很好","體驗很糟"]}'
```

## Train and evaluate (optional)

```bash
python training/train_sentiment.py --data data/labeled_seed.jsonl
python eval/eval_sentiment.py --data data/labeled_seed.jsonl
python benchmark/latency.py --texts-file data/labeled_seed.jsonl --batch 32
```

## Run Unit Tests

```bash
python -m unittest discover -s tests -v
```
