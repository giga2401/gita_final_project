# config.py
import json
import os

CONFIG_PATH = "config.json"

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from config file: {e}")
        return None
    except FileNotFoundError:
        print(f"Config file not found: {CONFIG_PATH}")
        return None
