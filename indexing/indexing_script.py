import os
import git
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
from config import load_config

class IndexingScript:
    def __init__(self):
        self.config = load_config()
        self.repo_dir = "repos"
        os.makedirs(self.repo_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.config["vector_db_path"])
        self.collection = self.client.get_or_create_collection(name=self.config["collection_name"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["embedding_model"])
        self.embedding_model = AutoModel.from_pretrained(self.config["embedding_model"])

    def fetch_repos(self):
        repositories = self.config["repositories"]
        if not repositories:
            print("No repositories found in config.")
            return

        for repo_url in repositories:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(self.repo_dir, repo_name)
            
            if not os.path.exists(repo_path):
                print(f"Cloning {repo_name} (shallow clone)...")
                try:
                    git.Repo.clone_from(repo_url, repo_path, depth=1, single_branch=True)
                except git.exc.GitCommandError as e:
                    print(f"Error cloning {repo_url}: {e}")
                    continue
            else:
                print(f"Repository {repo_name} already exists.")

            self.index_code_files(repo_path)

    def index_code_files(self, repo_path):
        # Get the list of programming languages from the config
        programming_languages = self.config.get("programming_languages")
        
        if not programming_languages:
            print("No programming languages specified in config. Skipping indexing.")
            return
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                # Check if the file extension is in the list of programming languages
                if any(file.endswith(lang) for lang in programming_languages):
                    file_path = os.path.join(root, file)
                    # Check if the file's embedding already exists
                    existing_embeddings = self.collection.get(ids=[file_path])
                    if existing_embeddings["ids"]:  # If the ID exists, skip this file
                        print(f"Skipping existing file: {file_path}")
                        continue

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        embedding = self.generate_embedding(code)
                        self.collection.add(
                            embeddings=[embedding],
                            metadatas=[{"file": file_path}],
                            ids=[file_path]
                        )

    def generate_embedding(self, code):
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding.tolist()

if __name__ == "__main__":
    script = IndexingScript()
    script.fetch_repos()