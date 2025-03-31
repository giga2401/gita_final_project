# Evaluation Service

## Overview

This service runs a Python script (`eval.py`) to evaluate the performance of the plagiarism detection system against a ground-truth dataset (`data/dataset.csv`). It compares three different approaches:

1.  **RAG + LLM:** The full system accessed via the `plagiarism_checker` API.
2.  **LLM Only:** Uses the configured LLM directly (via OpenAI API) with a specific prompt to assess plagiarism based only on the code snippet itself.
3.  **RAG Only:** Queries the vector database (ChromaDB) via the `embedding_server` for the closest match and makes a decision based on a similarity distance threshold (`rag_only_threshold`).

## Key Functionality

-   Loads the evaluation dataset (`data/dataset.csv`).
-   Iterates through each code snippet in the dataset.
-   Sends requests to the `plagiarism_checker` API (`/check_plagiarism`).
-   Sends requests to the `embedding_server` API (`/embed`) for RAG-only evaluation.
-   Queries the ChromaDB instance directly for RAG-only evaluation.
-   Calls the OpenAI API for LLM-only evaluation.
-   Compares the predictions from each approach against the expected labels.
-   Calculates standard classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
-   Saves detailed results and summary metrics to CSV files in the `evaluation_results/` directory.

## Dependencies

-   **Internal:**
    -   Requires the `plagiarism_checker` service to be running and healthy.
    -   Requires the `embedding_server` service to be running and healthy.
    -   Requires access to the populated ChromaDB database (`../chroma_db`).
    -   Reads configuration from `../data/config.json`.
    -   Uses code from `../shared`.
    -   Reads the dataset from `../data/dataset.csv`.
-   **External APIs:** OpenAI API (requires `OPENAI_API_KEY` in `.env`).
-   **External Libraries:** `pandas`, `scikit-learn`, `requests`, `openai`, `chromadb`, `python-dotenv`, `numpy`. See `requirements.txt`.
-   **Configuration:** Uses `plagiarism_api_port`, `embedding_api_port`, `llm_model`, `vector_db_path`, `collection_name`, `rag_only_threshold`, `dataset_path` from `config.json`. Reads `OPENAI_API_KEY` from `.env`.

## Setup & Installation

The service is intended to be built using Docker Compose and run as a one-off task.

1.  Ensure Docker and Docker Compose are installed.
2.  Ensure the `.env` file exists in the project root with a valid `OPENAI_API_KEY`.
3.  Build the image: `docker-compose build evaluation`

## Running the Evaluation

1.  Ensure the `embedding_server` and `plagiarism_checker` services are running (e.g., via `docker-compose up -d embedding_server plagiarism_checker`).
2.  Ensure the ChromaDB database (`./chroma_db`) has been populated by the `indexing` service.
3.  Run the evaluation script:
    ```bash
    docker-compose run --rm evaluation
    ```
    This command starts a new container, runs the `eval.py` script, and removes the container once finished. Results will be saved to the `./evaluation_results` directory on the host machine.

## Output

-   `evaluation_results/evaluation_results_comparison.csv`: Detailed results for each snippet across the three methods.
-   `evaluation_results/evaluation_metrics.csv`: Summary metrics (Accuracy, Precision, Recall, F1, Confusion Matrix) for each method.

## Key Files

-   `eval.py`: The main evaluation script.
-   `Dockerfile`: Instructions for building the Docker image.
-   `requirements.txt`: Python dependencies.
-   `../data/config.json`: Configuration source.
-   `../data/dataset.csv`: Evaluation data source.
-   `../shared/`: Contains shared configuration loading and utility functions.
-   `../evaluation_results/`: Directory where output files are saved (mounted as a volume).