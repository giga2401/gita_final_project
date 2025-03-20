from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
embedding_model = AutoModel.from_pretrained("microsoft/codebert-base")

class CodeSnippet(BaseModel):
    code: str

@app.post("/embed")
async def generate_embedding(snippet: CodeSnippet):
    try:
        # Tokenize the input code
        inputs = tokenizer(
            snippet.code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        if inputs['input_ids'].shape[1] >= 512:
            logging.warning("Code exceeds token limit and will be truncated.")

        # Generate the embedding
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return {"embedding": embedding.tolist()}
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)