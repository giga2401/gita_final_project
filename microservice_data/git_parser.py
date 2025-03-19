import os
import git
from code_chunker import CodeChunker
from config import load_config
import json

class GitRepoFetcher:
    def __init__(self, repo_dir="repos"):
        """Initializes with repo directory."""
        self.repo_dir = repo_dir
        os.makedirs(self.repo_dir, exist_ok=True)

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
            
            # Create an instance of CodeChunker to process code files and get embeddings
            chunker = CodeChunker()
            chunks = chunker.process_repository(repo_path)
            
            # Print confirmation of how many chunks (embeddings) have been stored
            print(f"Stored {len(chunks)} code embeddings in the vector database.")

    def store_chunks(self, chunks):
        """Stores or processes the code chunks as needed."""
        if not chunks:
            print("No chunks to store.")
            return

        output_file = "code_chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4)

        print(f"Stored {len(chunks)} chunks in {output_file}.")
