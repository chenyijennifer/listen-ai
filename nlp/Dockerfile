FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py lexicon.py ml_backend.py sentiment_service.py ./
COPY model_artifacts ./model_artifacts/

ENV NLP_PORT=8001
EXPOSE 8001

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${NLP_PORT}"]
