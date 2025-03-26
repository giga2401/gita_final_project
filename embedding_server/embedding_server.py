from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from shared.config import load_config
from shared.utils import setup_logging

setup_logging()

app = FastAPI()
config = load_config()

tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
embedding_model = AutoModel.from_pretrained(config["embedding_model"])

class CodeSnippet(BaseModel):
    code: str

@app.post("/embed")
async def generate_embedding(snippet: CodeSnippet):
    try:
        inputs = tokenizer(
            snippet.code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        if inputs['input_ids'].shape[1] >= 512:
            logging.warning("Code exceeds token limit and will be truncated.")

        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return {"embedding": embedding.tolist()}
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))