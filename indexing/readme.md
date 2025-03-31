# Indexing Service

## Overview

This service is responsible for populating the ChromaDB vector database with code embeddings. It performs two main tasks:

1.  **Fetching Repositories (`fetch_repos.py`):** Clones Git repositories specified in the `config.json` file into a local directory (`indexing/repos`).
2.  **Indexing Code (`index.py`):** Walks through the downloaded repositories, identifies code files based on specified language extensions, chunks the code, requests embeddings for the chunks from the `embedding_server`, and stores the embeddings, code text, and metadata in the ChromaDB collection.

## Key Functionality

-   **Repository Fetching:** Clones specified Git repositories using `GitPython`. Performs shallow clones for efficiency. Skips cloning if a repository directory already exists.
-   **File Discovery:** Scans the `indexing/repos` directory for files matching language extensions defined in `config.json` (e.g., `.py`, `.java`).
-   **Code Chunking:** Uses the tokenizer (corresponding to the embedding model) to split code files into smaller, potentially overlapping chunks suitable for embedding.
-   **Embedding Generation:** Sends batches of code chunks to the `embedding_server`'s `/embed_batch` endpoint to get their vector embeddings.
-   **Database Indexing:** Adds the embeddings, original code chunk text, and metadata (repository name, file path, chunk index) to the ChromaDB collection specified in `config.json`.
-   **Duplicate Prevention:** Optionally checks if a file has already been indexed (based on metadata) to avoid re-processing (controlled by `db_check_indexed_files` in `config.json`).

## Dependencies

-   **Internal:**
    -   Requires the `embedding_server` service to be running and healthy to generate embeddings.
    -   Reads configuration from `../data/config.json`.
    -   Uses code from `../shared`.
    -   Writes to the ChromaDB database (`../chroma_db`).
    -   Writes downloaded repositories to `../indexing/repos`.
-   **External Libraries:** `chromadb`, `transformers` (for tokenizer), `gitpython`, `requests`, `python-dotenv`, `numpy`, `sentencepiece`. See `requirements.txt`.
-   **Configuration:** Uses `repositories`, `programming_languages`, `embedding_model`, `embedding_api_port`, `vector_db_path`, `collection_name`, `max_chunk_length`, `chunk_stride`, `indexing_batch_size`, `db_check_indexed_files` from `config.json`.

## Setup & Installation

The service is intended to be built using Docker Compose. The scripts are run manually *inside* the container.

1.  Ensure Docker and Docker Compose are installed.
2.  Build the image: `docker-compose build indexing`

## Running the Indexing Process

The indexing service container runs `sleep infinity` by default, keeping it alive so you can execute commands within it.

1.  **Start the necessary services:**
    ```bash
    # Start embedding server (dependency) and the indexing container itself
    docker-compose up -d embedding_server indexing
    ```
    Wait for the `embedding_server` to become healthy (check `docker-compose ps`).

2.  **Fetch the repositories:**
    ```bash
    docker-compose exec indexing python indexing/fetch_repos.py
    ```
    This clones the repositories listed in `config.json` into the `/app/indexing/repos` directory inside the container (mounted from `./indexing/repos` on the host).

3.  **Run the indexing script:**
    ```bash
    docker-compose exec indexing python indexing/index.py
    ```
    This script will find code files in the fetched repos, chunk them, get embeddings from the `embedding_server`, and populate the ChromaDB database located at `/app/chroma_db` inside the container (mounted from `./chroma_db` on the host). This step can take a significant amount of time depending on the number and size of repositories.

## Key Files

-   `fetch_repos.py`: Script to clone Git repositories.
-   `index.py`: Script to process files, get embeddings, and index into ChromaDB.
-   `Dockerfile`: Instructions for building the Docker image.
-   `requirements.txt`: Python dependencies.
-   `../shared/`: Contains shared configuration loading and utility functions.
-   `../data/config.json`: Configuration source.
-   `../indexing/repos/`: Directory where repositories are cloned (volume mount).
-   `../chroma_db/`: Directory where the ChromaDB database is stored (volume mount).