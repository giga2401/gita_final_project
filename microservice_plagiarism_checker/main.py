from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
from config import load_config, get_openai_api_key
from prompt_templates import PLAGIARISM_CHECK_PROMPT

app = FastAPI()

# Load configuration
config = load_config()
EMBEDDING_MODEL = config.get("embedding_model", "microsoft/codebert-base")
LLM_MODEL = config.get("llm_model", "gpt-3.5-turbo")
VECTOR_DB_PATH = config.get("vector_db_path", "chroma_db")
COLLECTION_NAME = config.get("collection_name", "code_embeddings")

# Initialize OpenAI client
openai_api_key = get_openai_api_key()
if not openai_api_key:
    raise ValueError("OpenAI API key is missing.")
openai_client = OpenAI(api_key=openai_api_key)

# Initialize Hugging Face model for embeddings
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

class CodeSnippet(BaseModel):
    code: str

def generate_embedding(code: str):
    """Generates embeddings for the code using Hugging Face model."""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()

@app.post("/check_plagiarism")
async def check_plagiarism(snippet: CodeSnippet):
    """Checks if the submitted code is plagiarized."""
    try:
        # Generate embedding for the submitted code
        embedding = generate_embedding(snippet.code)

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
            model=LLM_MODEL,
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