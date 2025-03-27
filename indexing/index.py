import os
import sys
import logging
import requests
import time
from typing import List, Optional 

try:
    from shared.config import load_config
    from shared.utils import setup_logging
except ModuleNotFoundError as e:
    print(f"[Error] Could not import from 'shared'. Ensure the project root is in your PYTHONPATH.")
    print(f"  Current sys.path: {sys.path}")
    raise e

import chromadb
from transformers import AutoTokenizer

# --- Setups Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# Determines project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError: # Handles case where __file__ is not defined
    project_root = os.path.abspath('.')
logger.info(f"Project root determined as: {project_root}")

# --- Variable to detect Docker environment ---
IS_DOCKER_ENV = os.environ.get('IS_DOCKER_ENV', 'false').lower() == 'true'
logger.info(f"Running in Docker environment: {IS_DOCKER_ENV}")

class IndexProcessor:
    def __init__(self):
        logger.info("Initializing IndexProcessor...")
        try:
            self.config = load_config()
            self.repo_dir = os.path.join(project_root, "indexing", "repos")
            os.makedirs(self.repo_dir, exist_ok=True)

            # --- ChromaDB Client Setup ---
            db_path_from_config = self.config["vector_db_path"]
            if not os.path.isabs(db_path_from_config):
                 db_path = os.path.join(project_root, db_path_from_config)
            else:
                 db_path = db_path_from_config
            logger.info(f"Initializing ChromaDB client at resolved path: {db_path}")
            db_parent_dir = os.path.dirname(db_path)
            if db_parent_dir: os.makedirs(db_parent_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)

            self.collection_name = self.config["collection_name"]
            logger.info(f"Getting or creating collection: {self.collection_name}")
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' ready. Item count: {self.collection.count()}")

            # --- Tokenizer Setup ---
            model_name = self.config["embedding_model"]
            logger.info(f"Loading tokenizer (only): {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # --- Gets Embedding Service URL ---
            default_embedding_host = "embedding_server" if IS_DOCKER_ENV else "localhost"
            embedding_port = self.config.get("embedding_api_port", 8001)
            self.embedding_batch_url = os.environ.get("EMBEDDING_SERVICE_URL", f"http://{default_embedding_host}:{embedding_port}") + "/embed_batch"
            logger.info(f"Using Embedding Service Batch URL: {self.embedding_batch_url}")

            # --- Chunking Configuration ---
            self.max_chunk_length = self.config.get("max_chunk_length", 512)
            self.stride = self.config.get("chunk_stride", 256) 
            model_max_len = getattr(self.tokenizer, 'model_max_length', self.max_chunk_length)
            if self.max_chunk_length > model_max_len:
                 logger.warning(f"Configured max_chunk_length ({self.max_chunk_length}) exceeds tokenizer's model_max_length ({model_max_len}). Using {model_max_len}.")
                 self.max_chunk_length = model_max_len
            if self.stride >= self.max_chunk_length:
                 logger.warning(f"Chunk stride ({self.stride}) >= max_chunk_length ({self.max_chunk_length}). Adjusting stride.")
                 self.stride = int(self.max_chunk_length * 0.75)
            logger.info(f"Chunking config: max_length={self.max_chunk_length}, stride={self.stride}")

            # --- Indexing Configuration ---
            self.batch_size = self.config.get("indexing_batch_size", 32)
            self.check_indexed = self.config.get("db_check_indexed_files", True)

        except (RuntimeError, ValueError, Exception) as e:
             logger.critical(f"Failed to initialize IndexProcessor: {e}", exc_info=True)
             raise

    def chunk_code_tokens(self, code: str) -> List[List[int]]:
        """Tokenizes code and returns overlapping chunks as lists of token IDs."""
        if not code or not code.strip():
            return []
        try:
            inputs = self.tokenizer(
                code,
                max_length=self.max_chunk_length,
                stride=self.stride,
                truncation=True,
                return_overflowing_tokens=True,
                padding=False, # Because padding happens server-side for batches
                return_attention_mask=False 
            )
            # Checks if 'input_ids' and 'overflow_to_sample_mapping' are present
            input_ids_list = inputs.get('input_ids')
            if not input_ids_list:
                # If no overflowing tokens, the original input is one chunk (if it fits)
                single_input = self.tokenizer(code, truncation=True, max_length=self.max_chunk_length)
                return single_input.get('input_ids', [[]]) # Returns as list of lists

            return input_ids_list
        except Exception as e:
            logger.error(f"Error during tokenization/chunking: {e}", exc_info=False)
            return []

    def get_embeddings_batch_via_service(self, texts: List[str], retries=2, delay=3) -> Optional[List[List[float]]]:
        """Gets embeddings for a batch of texts by calling the embedding service with retries."""
        if not texts:
            return []

        for attempt in range(retries + 1):
            try:
                response = requests.post(self.embedding_batch_url, json={"texts": texts}, timeout=60) # Longer timeout for batch
                response.raise_for_status()
                result = response.json()
                if "embeddings" in result and isinstance(result["embeddings"], list) and len(result["embeddings"]) == len(texts):
                    return result["embeddings"]
                else:
                    logger.error(f"Invalid response format or length mismatch from embedding service batch endpoint: {result}")
                    return None
            except requests.exceptions.Timeout:
                logger.warning(f"Embedding service batch request timed out (Attempt {attempt+1}/{retries+1}). Retrying in {delay}s...")
            except requests.exceptions.RequestException as req_err:
                status_code = req_err.response.status_code if req_err.response is not None else None
                logger.warning(f"Embedding service batch request failed (Attempt {attempt+1}/{retries+1}): {req_err}. Status: {status_code}. Retrying in {delay}s...")
            except Exception as e:
                 logger.error(f"Unexpected error during embedding service batch call (Attempt {attempt+1}/{retries+1}): {e}", exc_info=True)

            if attempt < retries:
                 time.sleep(delay * (attempt + 1)) 
            else:
                 logger.error(f"Failed to get embeddings for batch after {retries+1} attempts.")
                 return None
        return None # unreachable if loop finishes

    def is_file_indexed(self, relative_path: str) -> bool:
        """Checks if a file with the given relative path has any chunks in the DB."""
        if not self.check_indexed:
            return False
        try:
            results = self.collection.get(
                where={"relative_path": relative_path},
                limit=1,
                include=[] # Doesn't need content, just checks existence
            )
            is_present = bool(results and results.get("ids"))
            if is_present:
                logger.debug(f"File '{relative_path}' found in DB, skipping.")
            return is_present
        except Exception as e:
            logger.error(f"Error checking index status for {relative_path}: {e}. Assuming not indexed.")
            return False

    def index_code_files(self):
        """Walks through repo directories and indexes code files using the embedding service."""
        programming_languages = self.config.get("programming_languages")
        if not programming_languages:
            logger.warning("No programming languages specified. Skipping indexing.")
            return

        logger.info(f"Starting indexing for languages: {programming_languages} in dir: {self.repo_dir}")
        stats = {"indexed_files": 0, "skipped_already_indexed": 0, "processed_chunks": 0, "error_files": 0, "embedding_errors": 0}

        files_to_process = []
        for root, _, files in os.walk(self.repo_dir):
            if ".git" in root.split(os.sep): continue

            for file in files:
                if any(file.lower().endswith(lang.lower()) for lang in programming_languages):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_dir)
                    files_to_process.append((file_path, relative_path))

        logger.info(f"Found {len(files_to_process)} potential files to index.")

        for file_path, relative_path in files_to_process:
            repo_name = relative_path.split(os.sep)[0] if os.sep in relative_path else "root"
            file_id_base = relative_path.replace(os.sep, "__").replace('.', '_') # Safer ID

            try:
                # --- Checks if already indexed ---
                if self.is_file_indexed(relative_path):
                    stats["skipped_already_indexed"] += 1
                    continue

                logger.info(f"Processing file: {relative_path}")

                # --- Reads File Content ---
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                    if not code or not code.strip():
                        logger.warning(f"Skipping empty file: {relative_path}")
                        continue
                except Exception as read_err:
                     logger.error(f"Error reading file {relative_path}: {read_err}")
                     stats["error_files"] += 1
                     continue

                # --- Chunks Code ---
                tokenized_chunks = self.chunk_code_tokens(code)
                if not tokenized_chunks:
                     logger.warning(f"No tokenized chunks generated for file: {relative_path}")
                     continue

                # --- Prepares Data for ChromaDB (Batch Process) ---
                chunks_data = [] # List of tuples: (chunk_id, chunk_text, metadata)
                for i, chunk_ids_list in enumerate(tokenized_chunks):
                    chunk_id = f"{file_id_base}_chunk_{i}"
                    try:
                        chunk_text = self.tokenizer.decode(chunk_ids_list, skip_special_tokens=True).strip()
                        if not chunk_text:
                            logger.warning(f"Skipping empty decoded chunk {i} for {relative_path}")
                            continue

                        chunk_metadata = {
                            "relative_path": relative_path,
                            "repo_name": repo_name,
                            "chunk_index": i,
                            "file_id_base": file_id_base # Stores base ID for reference
                        }
                        chunks_data.append((chunk_id, chunk_text, chunk_metadata))
                    except Exception as decode_err:
                        logger.error(f"Error decoding chunk {i} for {relative_path}: {decode_err}")
                        continue # Skips this chunk

                if not chunks_data:
                    logger.warning(f"No valid chunks prepared for file {relative_path}.")
                    continue

                # --- Gets Embeddings and Adds to ChromaDB in Batches ---
                added_chunk_count = 0
                file_had_embedding_error = False
                for i in range(0, len(chunks_data), self.batch_size):
                    batch = chunks_data[i : i + self.batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_texts = [item[1] for item in batch]
                    batch_metadatas = [item[2] for item in batch]

                    logger.debug(f"Getting embeddings for batch {i//self.batch_size + 1} ({len(batch_texts)} chunks) for {relative_path}...")
                    batch_embeddings = self.get_embeddings_batch_via_service(batch_texts)

                    if batch_embeddings and len(batch_embeddings) == len(batch_ids):
                        try:
                            self.collection.add(
                                embeddings=batch_embeddings,
                                metadatas=batch_metadatas,
                                documents=batch_texts, # Stores the text corresponding to the embedding
                                ids=batch_ids
                            )
                            added_chunk_count += len(batch_ids)

                        except Exception as add_err:
                            logger.error(f"Failed to add batch for {relative_path} to ChromaDB: {add_err}", exc_info=True)
                            stats["error_files"] += 1 
                            file_had_embedding_error = True 
                            break 
                    else:
                        logger.error(f"Failed to get embeddings for batch {i//self.batch_size + 1} of {relative_path}. Skipping batch.")
                        stats["embedding_errors"] += len(batch_ids)
                        file_had_embedding_error = True

                # --- Updates Stats for the File ---
                if added_chunk_count > 0:
                    stats["indexed_files"] += 1
                    stats["processed_chunks"] += added_chunk_count
                    if file_had_embedding_error:
                         logger.warning(f"File {relative_path} partially indexed due to errors.")
                elif file_had_embedding_error: # No chunks added, but there were errors
                    if stats["error_files"] == 0: # Avoids double counting if read error already happened
                       stats["error_files"] += 1
                    logger.error(f"No chunks successfully indexed for file {relative_path} due to errors.")
                else: # No chunks added, no errors reported (e.g., all chunks empty after decode)
                    logger.warning(f"No chunks were added to DB for file {relative_path} (possibly all empty).")

            except Exception as file_proc_err:
                logger.error(f"Unhandled error processing file {relative_path}: {file_proc_err}", exc_info=True)
                stats["error_files"] += 1
                continue # Moves to the next file

        logger.info(f"Indexing complete. Summary: {stats}")
        final_count = self.collection.count()
        logger.info(f"Final collection size: {final_count} items.")


if __name__ == "__main__":
    try:
        processor = IndexProcessor()
        processor.index_code_files()
        logger.info("Index processing finished successfully.")
    except Exception as main_err:
        logger.critical(f"An critical error occurred during IndexProcessor execution: {main_err}", exc_info=True)
        sys.exit(1)