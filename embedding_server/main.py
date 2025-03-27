from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import sys
import time
from typing import List 
from shared.config import load_config
from shared.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

try:
    config = load_config()
    MODEL_NAME = config['embedding_model']
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model.to(device)
    embedding_model.eval()
    logger.info(f"Embedding model loaded on device: {device}")

except Exception as E:
    logger.critical(f"Failed to initialize embedding server dependencies: {E}", exc_info=True)
    sys.exit(1)

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str

class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=128) # Add constraints

class EmbedResponse(BaseModel):
    embedding: List[float]

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]

def get_embedding(text: str) -> List[float]:
    """Generates embedding for a single text."""
    if not text or not text.strip():
        # Handles empty text case specifically if needed, or rely on tokenizer
        logger.warning("Attempting to embed empty or whitespace-only text.")
        pass

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True, # Pad to max length of batch or model max length
        max_length=tokenizer.model_max_length, # Ensures truncation
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = embedding_model(**inputs)

    # Mean pooling of the last hidden state
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    return embedding.tolist()

@app.post("/embed", response_model=EmbedResponse)
async def create_embedding_endpoint(request: EmbedRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    start_time = time.time()
    try:
        embedding = get_embedding(request.text)
        process_time = time.time() - start_time
        logger.info(f"Generated single embedding for text length {len(request.text)} in {process_time:.4f}s")
        return EmbedResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Error generating single embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")

@app.post("/embed_batch", response_model=EmbedBatchResponse)
async def create_embedding_batch_endpoint(request: EmbedBatchRequest):
    start_time = time.time()
    logger.info(f"Received batch embedding request for {len(request.texts)} texts.")
    if not request.texts:
         raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

    embeddings_list = []
    try:
        # Processes texts in batches using the tokenizer's batch encoding
        inputs = tokenizer(
            request.texts,
            return_tensors="pt",
            truncation=True,
            padding=True, # Pad to the longest sequence in the batch
            max_length=tokenizer.model_max_length,
            return_attention_mask=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = embedding_model(**inputs)

        # Mean pooling for each item in the batch
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings_list = batch_embeddings.tolist()

        process_time = time.time() - start_time
        logger.info(f"Generated {len(embeddings_list)} embeddings in batch in {process_time:.4f}s")
        return EmbedBatchResponse(embeddings=embeddings_list)

    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate batch embeddings: {e}")


@app.get("/health")
def health_check():
    try:
        assert embedding_model is not None
        assert tokenizer is not None
        return {"status": "Embedding Service is running and model appears loaded"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Returns 503 Service Unavailable if critical components aren't ready
        raise HTTPException(status_code=503, detail="Service dependencies not ready")


# Uvicorn Entry Point (for local run)
if __name__ == "__main__":
    import uvicorn
    server_port = int(config.get("embedding_api_port", 8001))
    logger.info(f"Starting Uvicorn server locally on port {server_port} with reload enabled.")
    # Uses "embedding_server.main:app" for running with python embedding_server/main.py
    uvicorn.run("embedding_server.main:app", host="0.0.0.0", port=server_port, reload=True)