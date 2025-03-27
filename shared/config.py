import json
import os
from dotenv import load_dotenv
import logging 

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

default_config_path = os.path.join(project_root, 'data', 'config.json')

CONFIG_PATH = os.environ.get("CONFIG_PATH", default_config_path)

def load_config():
    """Loads configuration from JSON file."""
    try:
        logging.info(f"Attempting to load config from: {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
             raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            required_keys = ["embedding_model", "llm_model", "vector_db_path", "collection_name"]
            if not all(key in config for key in required_keys):
                raise ValueError(f"Config missing one or more required keys: {required_keys}")
            logging.info("Configuration loaded successfully.")
            return config
    except (json.JSONDecodeError, FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Error loading or validating config file: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load or validate configuration: {e}")

def get_openai_api_key():
    """Retrieves OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables (.env file).")
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return api_key