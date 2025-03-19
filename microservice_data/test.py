from config import load_config, get_openai_api_key

# Test loading repository URLs from config.json
repositories = load_config()
print("Repositories:", repositories)

# Test retrieving OpenAI API Key from .env
api_key = get_openai_api_key()
print("OpenAI API Key:", api_key if api_key else "Not found")
