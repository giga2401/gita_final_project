# Multi-stage build to reduce image size
FROM python:3.9-slim as builder
RUN apt-get update && apt-get install -y git
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /app /app
WORKDIR /app
CMD ["python", "indexing_script.py"]