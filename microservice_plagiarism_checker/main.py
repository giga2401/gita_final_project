from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from config import load_config, get_openai_api_key
from openai import OpenAI
from prompt import PLAGIARISM_CHECK_PROMPT

app = FastAPI()
config = load_config()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=config["vector_db_path"])
collection = chroma_client.get_collection(name=config["collection_name"])

# Initialize OpenAI client
openai_api_key = get_openai_api_key()
if not openai_api_key:
    raise ValueError("OpenAI API key is missing.")
openai_client = OpenAI(api_key=openai_api_key)

class CodeSnippet(BaseModel):
    code: str

@app.post("/check_plagiarism")
async def check_plagiarism(snippet: CodeSnippet):
    try:
        # Generate embedding for the submitted code
        embedding = await get_embedding(snippet.code)

        # Search the vector database for similar code
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5  # Retrieve top 5 similar code snippets
        )

        # Prepare context for the LLM
        similar_files = results["metadatas"][0]
        context = "\n".join([f"File: {file['file']}\nCode: {file['summary']}" for file in similar_files])

        # Generate LLM prompt
        prompt = PLAGIARISM_CHECK_PROMPT.format(
            user_code=snippet.code,
            context=context
        )

        # Query the LLM
        response = openai_client.chat.completions.create(
            model=config["llm_model"],
            messages=[
                {"role": "system", "content": "You are a plagiarism detection system."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the LLM's response
        llm_response = response.choices[0].message.content.strip().lower()
        is_plagiarized = llm_response == "yes"

        # Prepare the response
        response_data = {
            "is_plagiarized": is_plagiarized,
            "references": similar_files if is_plagiarized else []
        }
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_embedding(code: str):
    """Gets embedding for the provided code."""
    import requests
    response = requests.post(
        f"http://localhost:{config['embedding_server_port']}/embed",
        json={"code": code}
    )
    return response.json()["embedding"]