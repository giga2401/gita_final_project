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