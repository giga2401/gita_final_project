import os
import sys
from shared.config import load_config
from shared.utils import setup_logging
import git
import logging

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.append(project_root)

setup_logging()

class RepoFetcher:
    def __init__(self):
        self.config = load_config()
        # Save repositories inside the indexing folder
        self.repo_dir = os.path.join("indexing", "repos")
        os.makedirs(self.repo_dir, exist_ok=True)

    def fetch_repos(self):
        repositories = self.config["repositories"]
        if not repositories:
            logging.warning("No repositories found in config.")
            return

        for repo_url in repositories:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(self.repo_dir, repo_name)

            if not os.path.exists(repo_path):
                logging.info(f"Cloning {repo_name} (shallow clone)...")
                try:
                    git.Repo.clone_from(repo_url, repo_path, depth=1, single_branch=True)
                except git.exc.GitCommandError as e:
                    logging.error(f"Error cloning {repo_url}: {e}")
                    continue
            else:
                logging.info(f"Repository {repo_name} already exists.")

if __name__ == "__main__":
    fetcher = RepoFetcher()
    fetcher.fetch_repos()