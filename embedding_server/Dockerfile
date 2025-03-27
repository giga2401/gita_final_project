FROM python:3.9-slim

# Installing curl for health checks
RUN apt-get update && apt-get install -y curl --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./shared /app/shared
COPY embedding_server/requirements.txt .

# Upgrades pip
RUN pip install --no-cache-dir --upgrade pip

# command-line flags for index URLs
RUN pip install \
    --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY ./embedding_server /app/embedding_server

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PORT=8001
ENV IS_DOCKER_ENV=true

EXPOSE ${PORT}

CMD uvicorn embedding_server.main:app --host 0.0.0.0 --port ${PORT}