# Code Plagiarism Detector

## Overview

This project implements a multi-stage system to detect potential plagiarism in code snippets. It utilizes a Retrieval-Augmented Generation (RAG) approach, combining a vector database of code embeddings with a Large Language Model (LLM) for analysis and decision-making. The system is built as a set of microservices orchestrated using Docker Compose.

## Architecture

The system consists of the following core microservices:

1.  **Embedding Server (`embedding_server`):** A FastAPI service that generates vector embeddings for code snippets using a transformer model (e.g., CodeBERT).
2.  **Indexing Service (`indexing`):** A set of scripts responsible for fetching code from specified Git repositories, chunking the code files, generating embeddings via the `embedding_server`, and storing them in a ChromaDB vector database.
3.  **Plagiarism Checker API (`plag_checker`):** The main FastAPI service. It receives a code snippet, retrieves its embedding, queries the ChromaDB for similar code, and uses an LLM (OpenAI's GPT) with the retrieved context to determine if the code is plagiarized.
4.  **Evaluation Service (`evaluation`):** A script to evaluate the plagiarism checker's performance against a labeled dataset (`data/dataset.csv`), comparing the full RAG+LLM approach with LLM-only and RAG-only methods.
5.  **Shared Utilities (`shared`):** Common Python code for configuration loading and logging used by the other services.

**Key Technologies:**

-   Docker & Docker Compose
-   Python 3.9
-   FastAPI & Uvicorn (for APIs)
-   ChromaDB (Vector Database)
-   Transformers (Hugging Face library for embedding models)
-   PyTorch (Backend for Transformers)
-   OpenAI API (for the LLM component)
-   GitPython (for repository cloning)
-   Pandas & Scikit-learn (for evaluation)

## Features

-   Fetches and indexes code from multiple Git repositories.
-   Provides an API endpoint (`/check_plagiarism`) to check code snippets.
-   Utilizes RAG to provide relevant context to the LLM.
-   Configurable embedding models, LLM models, and thresholds via `config.json`.
-   Includes an evaluation pipeline to measure performance.
-   Containerized using Docker for easy setup and deployment.

## Project Structure

gita_final_project/
├── .env # Stores secrets like API keys (Needs to be created)
├── .gitignore
├── data/
│ ├── config.json # Main configuration file
│ └── dataset.csv # Evaluation dataset
├── docker-compose.yml # Docker Compose configuration
├── embedding_server/ # Embedding generation API service
│ ├── Dockerfile
│ ├── main.py
│ ├── requirements.txt
│ └── readme.md
├── evaluation/ # Evaluation script service
│ ├── Dockerfile
│ ├── eval.py
│ ├── requirements.txt
│ └── readme.md
├── evaluation_results/ # Output directory for evaluation (Created by evaluation service)
├── indexing/ # Repository fetching and indexing service
│ ├── Dockerfile
│ ├── fetch_repos.py
│ ├── index.py
│ ├── requirements.txt
│ └── readme.md
│ └── repos/ # Cloned repositories (Created by indexing service)
├── plag_checker/ # Main plagiarism check API service
│ ├── Dockerfile
│ ├── main.py
│ ├── prompt.py
│ ├── requirements.txt
│ └── readme.md
├── shared/ # Shared utility code
│ ├── config.py
│ ├── utils.py
│ └── readme.md
├── chroma_db/ # Vector database storage (Created by indexing service)
├── README.md # This file
└── setup.py # Basic Python package setup (Optional)

## Setup

**Prerequisites:**

-   Docker ([Install Docker](https://docs.docker.com/engine/install/))
-   Docker Compose ([Install Docker Compose](https://docs.docker.com/compose/install/))
-   Git (for cloning this repository)
-   OpenAI API Key ([Get an API Key](https://platform.openai.com/account/api-keys))

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd gita_final_project
    ```

2.  **Create the environment file:**
    Copy the example or create a new `.env` file in the project root:
    ```env
    # .env
    OPENAI_API_KEY=sk-YourSecretOpenAiApiKeyHere
    # Optionally override the default plagiarism API port (8000)
    # PLAGIARISM_API_PORT=8000
    ```
    Replace `sk-YourSecretOpenAiApiKeyHere` with your actual OpenAI API key.

3.  **Review Configuration:**
    Check `data/config.json` and adjust parameters if needed (e.g., `repositories`, `embedding_model`, `llm_model`).

4.  **Build Docker Images:**
    ```bash
    docker-compose build
    ```
    This might take some time, especially for the `embedding_server` which downloads the model.

## Running the System (Workflow)

1.  **Start Core Services:**
    Start the embedding server and the main plagiarism checker API in detached mode:
    ```bash
    docker-compose up -d embedding_server plagiarism_checker
    ```
    Wait for them to become healthy (check `docker-compose ps`). The `embedding_server` might take a few minutes on the first start to download the model.

2.  **Fetch Repositories:**
    Execute the repository fetching script *inside* the indexing container (which needs to be running, `docker-compose up -d indexing` if not already started alongside others, or just run the `exec` command):
    ```bash
    docker-compose exec indexing python indexing/fetch_repos.py
    ```
    This clones the repositories listed in `data/config.json` into the `./indexing/repos/` directory.

3.  **Index Code:**
    Execute the indexing script *inside* the indexing container:
    ```bash
    docker-compose exec indexing python indexing/index.py
    ```
    This is the most time-consuming step. It processes files, gets embeddings from the `embedding_server`, and populates the `./chroma_db/` vector database. Monitor the logs using `docker-compose logs -f indexing`.

4.  **Check Plagiarism:**
    Once indexing is complete, you can send requests to the plagiarism checker API (running on port 8000 by default):
    ```bash
    curl -X POST http://localhost:8000/check_plagiarism \
         -H "Content-Type: application/json" \
         -d '{
               "code": "def main():\n    print(\"Hello, World!\")\n\nmain()"
             }'
    ```
    Replace the code snippet with the one you want to check.

5.  **Run Evaluation (Optional):**
    To evaluate the system against the `data/dataset.csv`:
    ```bash
    docker-compose run --rm evaluation
    ```
    Ensure `plagiarism_checker` and `embedding_server` are running. Results will appear in `./evaluation_results/`.

6.  **Stop Services:**
    To stop all running services and remove the containers:
    ```bash
    docker-compose down
    ```
    To stop without removing containers:
    ```bash
    docker-compose stop
    ```

## Configuration

-   **`data/config.json`:** Controls model names, database paths, API ports, thresholds, repositories to index, chunking parameters, etc. See comments within the file or the `shared/config.py` loader for details.
-   **`.env`:** Stores secrets, primarily the `OPENAI_API_KEY`. Can also override the host port for the plagiarism API via `PLAGIARISM_API_PORT`.

## Services Quick Reference

-   **`embedding_server`:** Generates code embeddings. API at `http://embedding_server:8001` (internal). See `embedding_server/readme.md`.
-   **`indexing`:** Fetches repos and populates DB. Run via `docker-compose exec`. See `indexing/readme.md`.
-   **`plag_checker`:** Main plagiarism detection API. Exposed at `http://localhost:8000` (default). See `plag_checker/readme.md`.
-   **`evaluation`:** Runs evaluation script. Run via `docker-compose run`. See `evaluation/readme.md`.
-   **`shared`:** Utility code library. See `shared/readme.md`.

## Contributing / Future Work

*(Optional: Add guidelines for contributing or ideas for future improvements here)*
-   Support more programming languages.
-   Implement more sophisticated chunking strategies.
-   Explore different embedding and LLM models.
-   Add a simple web UI.
-   Improve error handling and resilience.