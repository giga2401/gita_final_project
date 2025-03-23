import json
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json'))

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise RuntimeError(f"Error loading config file: {e}")

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return api_key