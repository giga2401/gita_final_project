# Embedding Server

## Overview

This service provides a FastAPI-based API endpoint to generate text embeddings using a pre-trained transformer model (specified in `config.json`). It is designed to be used by other services within the project (like the Indexing and Plagiarism Checker) that require vector representations of code snippets.

## Key Functionality

-   Loads a transformer model (e.g., `microsoft/codebert-base`) and tokenizer for generating embeddings.
-   Provides a `/embed` endpoint to generate an embedding for a single piece of text.
-   Provides an `/embed_batch` endpoint for efficient generation of embeddings for multiple texts at once.
-   Includes a `/health` endpoint for health checks, ensuring the model is loaded and the service is operational.
-   Runs using Uvicorn within a Docker container.

## Dependencies

-   **Internal:** None directly, but serves embeddings to `indexing` and `plag_checker`. Reads configuration from `../data/config.json`. Uses code from `../shared`.
-   **External Libraries:** `fastapi`, `uvicorn`, `torch`, `transformers`, `python-dotenv`, `numpy`, `sentencepiece`, `requests`. See `requirements.txt`.
-   **Configuration:** Uses `embedding_model` and `embedding_api_port` from `config.json`.

## Setup & Installation

The service is intended to be built and run using Docker Compose.

1.  Ensure Docker and Docker Compose are installed.
2.  Build the image: `docker-compose build embedding_server`

## Running the Service

-   **Using Docker Compose (Recommended):**
    ```bash
    docker-compose up embedding_server
    ```
    The service will be available within the Docker network at `http://embedding_server:8001`.

-   **Local Development:**
    1.  Install requirements: `pip install -r embedding_server/requirements.txt` (preferably in a virtual environment).
    2.  Run the server:
        ```bash
        # Ensure PYTHONPATH includes the root directory if running from root
        export PYTHONPATH="${PYTHONPATH}:."
        uvicorn embedding_server.main:app --reload --port 8001
        ```

## API Endpoints

-   `POST /embed`:
    -   Request Body: `{ "text": "your code or text here" }`
    -   Response Body: `{ "embedding": [0.1, 0.2, ...] }`
-   `POST /embed_batch`:
    -   Request Body: `{ "texts": ["text one", "text two", ...] }` (Max 128 texts per batch)
    -   Response Body: `{ "embeddings": [[0.1, ...], [0.2, ...], ...] }`
-   `GET /health`:
    -   Response Body: `{ "status": "Embedding Service is running..." }` (or 503 if not ready)

## Key Files

-   `main.py`: The FastAPI application logic.
-   `Dockerfile`: Instructions for building the Docker image.
-   `requirements.txt`: Python dependencies.
-   `../shared/`: Contains shared configuration loading and utility functions.
-   `../data/config.json`: Service configuration is read from here.