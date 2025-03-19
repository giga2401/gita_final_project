import os
import json
import openai
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from config import get_openai_api_key

class CodeChunker:
    def __init__(self, model_name="sentence-transformers/codebert-base"):
        """Initializes the embedding model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.db = chromadb.PersistentClient(path="vector_db")
        self.collection = self.db.get_or_create_collection("code_chunks")

    def process_repository(self, repo_path):
        """Processes repository files and computes embeddings."""
        chunks = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".py", ".js", ".java", ".cpp")):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        embedding = self.compute_embedding(code)
                        chunks.append({"file": file_path, "embedding": embedding})
                        self.store_embedding(file_path, embedding)
        return chunks

    def compute_embedding(self, code):
        """Computes embedding vector for a code snippet."""
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Average pooling

    def store_embedding(self, file_path, embedding):
        """Stores embeddings in ChromaDB."""
        self.collection.add(
            documents=[file_path],
            embeddings=[embedding],
            ids=[file_path]
        )