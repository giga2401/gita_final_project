import json
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = "config.json"

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        return {}

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in .env file.")
    return api_key

def get_dataset_path():
    config = load_config()
    return config.get("dataset_path", "data/dataset.csv")