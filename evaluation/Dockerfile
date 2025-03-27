FROM python:3.9-slim

WORKDIR /app

COPY ./shared /app/shared

COPY evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./evaluation /app/evaluation
# Copys data needed at runtime (config, dataset)
COPY ./data/config.json /app/data/config.json
COPY ./data/dataset.csv /app/data/dataset.csv

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "evaluation/eval.py"]