from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from shared.config import load_config, get_openai_api_key
from openai import OpenAI
from prompt import PLAGIARISM_CHECK_PROMPT
import requests
import logging
from shared.utils import setup_logging

setup_logging()

app = FastAPI()
config = load_config()

chroma_client = chromadb.PersistentClient(path=config["vector_db_path"])
try:
    collection = chroma_client.get_collection(name=config["collection_name"])
except chromadb.errors.InvalidCollectionException:
    logging.warning(f"Collection {config['collection_name']} does not exist. Creating a new collection.")
    collection = chroma_client.create_collection(name=config["collection_name"])

openai_client = OpenAI(api_key=get_openai_api_key())

class CodeSnippet(BaseModel):
    code: str

@app.post("/check_plagiarism")
async def check_plagiarism(snippet: CodeSnippet):
    try:
        if not snippet.code or not snippet.code.strip():
            raise HTTPException(status_code=400, detail="Code snippet cannot be empty.")

        embedding = await get_embedding(snippet.code)
        results = collection.query(query_embeddings=[embedding], n_results=5)

        similar_files = results.get("metadatas", [[]])
        if not similar_files or not isinstance(similar_files[0], list):
            similar_files = [[]]
        similar_files = similar_files[0]

        context = "\n".join([f"File: {file.get('file', 'Unknown')}\nCode: {file.get('summary', 'No summary available')}" for file in similar_files])

        prompt = PLAGIARISM_CHECK_PROMPT.format(
            user_code=snippet.code,
            context=context
        )

        response = openai_client.chat.completions.create(
            model=config["llm_model"],
            messages=[
                {"role": "system", "content": "You are a plagiarism detection system."},
                {"role": "user", "content": prompt}
            ]
        )

        llm_response = response.choices[0].message.content.strip().lower()
        if llm_response not in ["yes", "no"]:
            logging.error(f"Invalid LLM response: {llm_response}")
            raise HTTPException(status_code=500, detail="LLM returned an invalid response.")
        is_plagiarized = llm_response == "yes"

        return {"is_plagiarized": is_plagiarized, "references": similar_files if is_plagiarized else []}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in plagiarism check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_embedding(code: str):
    try:
        if not code or not code.strip():
            raise ValueError("Code snippet cannot be empty.")

        response = requests.post(
            f"http://localhost:{config['embedding_server_port']}/embed",
            json={"code": code},
            timeout=10
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Embedding server error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding server error: {e}")
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))