# Shared Utilities

## Overview

This directory contains Python modules with utility functions and configuration handling logic that are shared across multiple services within the `gita_final_project`. This promotes code reuse and consistency.

## Functionality

-   **`config.py`:**
    -   `load_config()`: Loads the main project configuration from the JSON file specified by the `CONFIG_PATH` environment variable (defaults to `../data/config.json`). Performs basic validation for required keys, such as `database_url`, `api_key`, and `log_level`.
    -   `get_openai_api_key()`: Retrieves the `OPENAI_API_KEY` from environment variables (typically loaded from the `.env` file in the project root). Raises a `KeyError` if the key is not found.
-   **`utils.py`:**
    -   `setup_logging()`: Configures basic logging for the application using Python's standard `logging` module, directing output to standard output with a consistent format (e.g., `2023-03-15 14:23:01,123 - INFO - Message`).

## Dependencies

-   **External Libraries:** `python-dotenv` (used implicitly by `load_dotenv()` in `config.py` to load the `.env` file).
-   **Project Files:** Reads `../data/config.json` and potentially `../.env`.

## Usage

These modules are not run directly. They are imported by other services (`embedding_server`, `indexing`, `plag_checker`, `evaluation`).

Example Import:

# In another service's main.py or script
import sys
import os
# Adjust path if necessary, though Docker PYTHONPATH usually handles this
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

from shared.config import load_config, get_openai_api_key
from shared.utils import setup_logging

# Setup logging first
setup_logging()

# Load config and secrets
config = load_config()
api_key = get_openai_api_key()

## Key Files
config.py: Handles loading configuration and API keys.

utils.py: Provides logging setup utility.

---**