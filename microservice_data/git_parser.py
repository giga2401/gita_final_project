import os
import git
import chromadb
from chromadb.config import Settings
from code_chunker import CodeChunker
from config import load_config

class GitRepoFetcher:
    def __init__(self, repo_dir="repos"):
        """Initializes with repo directory and ChromaDB client."""
        self.repo_dir = repo_dir
        os.makedirs(self.repo_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name="code_embeddings")

    def fetch_repos(self):
        """Fetches repositories from GitHub and clones them with a shallow clone."""
        repositories = load_config()
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

            # Chunks the code after cloning
            chunker = CodeChunker()
            chunks = chunker.process_repository(repo_path)

            # Store embeddings in ChromaDB
            self.store_embeddings(chunks)

    def store_embeddings(self, chunks):
        """Stores embeddings in ChromaDB."""
        if not chunks:
            print("No chunks to store.")
            return

        for chunk in chunks:
            self.collection.add(
                embeddings=[chunk["embedding"]],
                metadatas=[{"file": chunk["file"], "summary": chunk["summary"]}],
                ids=[chunk["file"]]  # Use file path as ID
            )

        print(f"Stored {len(chunks)} embeddings in ChromaDB.")