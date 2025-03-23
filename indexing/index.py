import os
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from shared.config import load_config
from shared.utils import setup_logging
import sys

# Setting the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.append(project_root)

setup_logging()

class IndexProcessor:
    def __init__(self):
        self.config = load_config()
        self.repo_dir = os.path.join("indexing", "repos")
        os.makedirs(self.repo_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.config["vector_db_path"])
        self.collection = self.client.get_or_create_collection(name=self.config["collection_name"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["embedding_model"])
        self.embedding_model = AutoModel.from_pretrained(self.config["embedding_model"])

    def index_code_files(self):
        programming_languages = self.config.get("programming_languages")
        if not programming_languages:
            logging.warning("No programming languages specified in config. Skipping indexing.")
            return

        for repo_name in os.listdir(self.repo_dir):
            repo_path = os.path.join(self.repo_dir, repo_name)
            if os.path.isdir(repo_path):
                self._index_repo(repo_path, repo_name)

    def _index_repo(self, repo_path, repo_name):
        for root, _, files in os.walk(repo_path):
            for file in files:
                if any(file.endswith(lang) for lang in self.config["programming_languages"]):
                    file_path = os.path.join(root, file)
                    file_id = f"{repo_name}_{file_path}"
                    existing_embeddings = self.collection.get(ids=[file_id])
                    if existing_embeddings["ids"]:
                        logging.info(f"Skipping existing file: {file_path}")
                        continue

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        embedding = self.generate_embedding(code)
                        self.collection.add(
                            embeddings=[embedding],
                            metadatas=[{"file": file_path}],
                            ids=[file_id]
                        )

    def generate_embedding(self, code):
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        if inputs['input_ids'].shape[1] >= 512:
            logging.warning(f"Code exceeds token limit and will be truncated.")
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding.tolist()

if __name__ == "__main__":
    processor = IndexProcessor()
    processor.index_code_files()