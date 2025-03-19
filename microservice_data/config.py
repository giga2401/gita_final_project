import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG_PATH = "config.json"

def load_config():
    """Loads repository URLs from the config file."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("repositories", [])
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        return []

def get_openai_api_key():
    """Retrieves OpenAI API Key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in .env file.")
    return api_key