import os
import sys
import git
import logging

# --- Setups Paths and Logging ---

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    project_root = os.path.abspath('.')

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports shared modules after setting path
from shared.config import load_config
from shared.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class RepoFetcher:
    def __init__(self):
        logger.info("Initializing RepoFetcher...")
        try:
            self.config = load_config()
            # Defines repo dir, relative to project root
            self.repo_dir = os.path.join(project_root, "indexing", "repos")
            os.makedirs(self.repo_dir, exist_ok=True)
            logger.info(f"Target repository directory: {self.repo_dir}")
        except Exception as e:
            logger.critical(f"Failed to initialize RepoFetcher: {e}", exc_info=True)
            raise

    def fetch_repos(self):
        repositories = self.config.get("repositories", []) 
        if not repositories:
            logger.warning("No repositories found in config. Nothing to fetch.")
            return

        logger.info(f"Fetching/checking {len(repositories)} repositories...")
        fetched_count = 0
        skipped_count = 0
        error_count = 0

        for repo_url in repositories:
            try:
                # Basic validation of URL format
                if not repo_url.startswith(("http://", "https://")) or not repo_url.endswith(".git"):
                     if not repo_url.startswith(("http://", "https://")):
                        logger.warning(f"Skipping potentially invalid repo URL format: {repo_url}")

                repo_name = repo_url.split("/")[-1]
                if repo_name.endswith(".git"):
                    repo_name = repo_name[:-4] # Removes .git suffix

                if not repo_name: # Handles edge case of bad URL parse
                    logger.error(f"Could not determine repository name from URL: {repo_url}")
                    error_count += 1
                    continue

                repo_path = os.path.join(self.repo_dir, repo_name)

                if not os.path.exists(repo_path):
                    logger.info(f"Cloning '{repo_name}' from {repo_url} (shallow clone)...")
                    try:
                        # Shallow clone for faster download, only master/main branch
                        git.Repo.clone_from(repo_url, repo_path, depth=1, single_branch=True)
                        logger.info(f"Successfully cloned '{repo_name}'.")
                        fetched_count += 1
                    except git.exc.GitCommandError as e:
                        logger.error(f"Error cloning {repo_url}: {e}")
                        error_count += 1
                        # Cleans up partially cloned repo if it exists
                        if os.path.exists(repo_path):
                            try:
                                import shutil
                                shutil.rmtree(repo_path)
                                logger.info(f"Cleaned up failed clone directory: {repo_path}")
                            except Exception as rm_err:
                                logger.error(f"Error removing failed clone directory {repo_path}: {rm_err}")
                        continue # Moves to next repo on error
                else:
                    logger.info(f"Repository '{repo_name}' already exists at {repo_path}. Skipping clone.")
                    skipped_count += 1

            except Exception as outer_err:
                 logger.error(f"Unexpected error processing repository {repo_url}: {outer_err}", exc_info=True)
                 error_count += 1

        logger.info(f"Repository fetching complete. Fetched: {fetched_count}, Skipped: {skipped_count}, Errors: {error_count}")

if __name__ == "__main__":
    try:
        fetcher = RepoFetcher()
        fetcher.fetch_repos()
    except Exception as main_err:
        logger.critical(f"RepoFetcher failed: {main_err}", exc_info=True)
        sys.exit(1)