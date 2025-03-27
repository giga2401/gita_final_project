
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./shared /app/shared
COPY indexing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./indexing /app/indexing
COPY ./data/config.json /app/data/config.json
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["sleep", "infinity"]