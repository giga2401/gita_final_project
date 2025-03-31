# Plagiarism Checker API

## Overview

This is the main API service for the plagiarism detection system. It provides a FastAPI endpoint (`/check_plagiarism`) that accepts a code snippet and uses a Retrieval-Augmented Generation (RAG) approach combined with a Large Language Model (LLM) to determine if the code is likely plagiarized.

## Key Functionality

-   **API Endpoint:** Exposes a `POST /check_plagiarism` endpoint accepting JSON `{ "code": "user code snippet" }`.
-   **Embedding Retrieval:** Calls the `embedding_server` to get the vector embedding for the submitted code snippet.
-   **Vector Database Query (Retrieval):** Queries the ChromaDB vector database (populated by the `indexing` service) using the obtained embedding to find the most similar code chunks from the indexed repositories.
-   **Context Preparation:** Formats the retrieved similar code chunks (including metadata like repository and file path) into a context string.
-   **LLM Invocation (Generation):** Constructs a prompt (using `prompt.py`) containing the user's code and the retrieved context. Sends this prompt to the configured LLM (via OpenAI API) to get a plagiarism decision (`true`/`false`) and a brief reasoning.
-   **Early Exit (Optional):** If configured (`plagiarism_similarity_threshold` in `config.json`), it can return "not plagiarized" early if the closest match found in the database is below the similarity threshold (i.e., distance is too high).
-   **Response:** Returns a JSON response like `{ "is_plagiarized": boolean, "reasoning": "...", "references": [...] }`. References are included only if plagiarism is detected.
-   **Health Check:** Provides a basic `GET /` endpoint to check if the service is running and can connect to the database.

## Dependencies

-   **Internal:**
    -   Requires the `embedding_server` service to be running and healthy.
    -   Requires access to the populated ChromaDB database (`../chroma_db`).
    -   Reads configuration from `../data/config.json`.
    -   Reads the OpenAI API key from `../.env`.
    -   Uses code from `../shared`.
    -   Uses the prompt template from `prompt.py`.
-   **External APIs:** OpenAI API (requires `OPENAI_API_KEY` in `.env`).
-   **External Libraries:** `fastapi`, `uvicorn`, `chromadb`, `openai`, `requests`, `python-dotenv`, `numpy`. See `requirements.txt`.
-   **Configuration:** Uses `embedding_api_port`, `vector_db_path`, `collection_name`, `llm_model`, `rag_num_results`, `max_llm_context_length`, `plagiarism_similarity_threshold`, `plagiarism_api_port` from `config.json`. Reads `OPENAI_API_KEY` from `.env`.

## Setup & Installation

The service is intended to be built and run using Docker Compose.

1.  Ensure Docker and Docker Compose are installed.
2.  Ensure the `.env` file exists in the project root with a valid `OPENAI_API_KEY`.
3.  Build the image: `docker-compose build plagiarism_checker`

## Running the Service

-   **Using Docker Compose (Recommended):**
    ```bash
    docker-compose up plagiarism_checker
    ```
    The service will be available on the host machine at `http://localhost:8000` (or the port specified by `PLAGIARISM_API_PORT` in your environment/`.env`). It also depends on the `embedding_server`. Docker compose will typically start the dependency automatically.

-   **Local Development:**
    1.  Install requirements: `pip install -r plag_checker/requirements.txt` (preferably in a virtual environment).
    2.  Ensure `embedding_server` is running.
    3.  Ensure `chroma_db` directory exists and is populated.
    4.  Ensure `.env` file is present in the project root.
    5.  Run the server:
        ```bash
        # Ensure PYTHONPATH includes the root directory if running from root
        export PYTHONPATH="${PYTHONPATH}:."
        uvicorn plag_checker.main:app --reload --port 8000
        ```

## API Endpoints

-   `POST /check_plagiarism`:
    -   Request Body: `{ "code": "user code snippet here" }`
    -   Response Body (Example Plagiarized):
        ```json
        {
            "is_plagiarized": true,
            "reasoning": "The user code shares significant structural and logical similarities with a snippet found in repo 'CS50x-2022', file 'pset1/mario.c'.",
            "references": [
                {
                    "repo": "CS50x-2022",
                    "file": "pset1/mario.c",
                    "closest_chunk": 2,
                    "distance": 0.15
                }
            ]
        }
        ```
    -   Response Body (Example Not Plagiarized):
        ```json
        {
            "is_plagiarized": false,
            "reasoning": "No significantly similar code snippets were found in the reference database.",
            "references": []
        }
        ```
-   `GET /`: Basic health check. Returns service status and DB connection status.

## Key Files

-   `main.py`: The FastAPI application logic.
-   `prompt.py`: Contains the LLM prompt template.
-   `Dockerfile`: Instructions for building the Docker image.
-   `requirements.txt`: Python dependencies.
-   `../shared/`: Contains shared configuration loading and utility functions.
-   `../data/config.json`: Service configuration is read from here.
-   `../.env`: OpenAI API key is read from here.
-   `../chroma_db/`: Path to the vector database (mounted as a volume).