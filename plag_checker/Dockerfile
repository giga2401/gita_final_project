FROM python:3.9-slim

# Installs curl for health checks
RUN apt-get update && apt-get install -y curl --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./shared /app/shared
COPY plag_checker/requirements.txt .

# Installs dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code AFTER installing dependencies
COPY ./plag_checker /app/plag_checker
COPY ./data/config.json /app/data/config.json

EXPOSE 8000

# Sets environment variables
ENV PORT=8000
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV IS_DOCKER_ENV=true

# Command to run the application
CMD uvicorn plag_checker.main:app --host 0.0.0.0 --port ${PORT}