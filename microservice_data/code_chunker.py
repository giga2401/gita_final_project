import os
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
from config import get_openai_api_key

class CodeChunker:
    def __init__(self, model_name="gpt-3.5-turbo", embedding_model_name="microsoft/codebert-base", max_tokens=16000):
        """Initializes OpenAI model and Hugging Face embedding model."""
        self.api_key = get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key is missing.")
        
        # Initialize OpenAI client (new API)
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens  # Maximum tokens allowed by the model
        
        # Initialize Hugging Face model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

        # Maximum file size to process (1 MB)
        self.MAX_FILE_SIZE = 1024 * 1024  # 1 MB

    def process_repository(self, repo_path):
        """Processes repository files and returns code chunks with embeddings."""
        chunks = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".py", ".js", ".java", ".cpp")):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    if file_size > self.MAX_FILE_SIZE:
                        print(f"Skipping large file: {file_path} ({file_size} bytes)")
                        continue
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        code_chunks = self.split_code_into_chunks(code)
                        for chunk in code_chunks:
                            try:
                                summary = self.chunk_code(chunk)
                                embedding = self.generate_embedding(chunk)
                                chunks.append({"file": file_path, "summary": summary, "embedding": embedding})
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
        return chunks

    def split_code_into_chunks(self, code):
        """Splits the code into smaller chunks based on the max token limit."""
        lines = code.splitlines()
        chunks = []
        current_chunk = []
        current_token_count = 0

        for line in lines:
            line_token_count = len(self.tokenizer.tokenize(line))
            if current_token_count + line_token_count > self.max_tokens:
                if current_chunk:  # Avoid empty chunks
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_token_count = line_token_count
            else:
                current_chunk.append(line)
                current_token_count += line_token_count

        if current_chunk:  # Add the last chunk
            chunks.append("\n".join(current_chunk))

        return chunks

    def chunk_code(self, code):
        """Chunks the code and summarizes it using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Summarize this code snippet."},
                {"role": "user", "content": code}
            ]
        )
        summary = response.choices[0].message.content
        return summary

    def generate_embedding(self, code):
        """Generates embeddings for the code using Hugging Face model."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,  # Explicitly truncate to max_length
            padding=True,
            max_length=512     # Ensure the input is truncated to 512 tokens
        )
        token_count = inputs["input_ids"].shape[1]
        if token_count > 512:
            print(f"Warning: Token count ({token_count}) exceeds model limit. Truncating.")
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding