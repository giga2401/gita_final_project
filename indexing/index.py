import os
import chromadb
# from chromadb.config import Settings # Not strictly needed for client usage
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from shared.config import load_config
from shared.utils import setup_logging
import sys

# Setting the working directory to the project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError: # Handle case where __file__ is not defined (e.g., interactive session)
    project_root = os.path.abspath('.') # Assume current directory is project root

os.chdir(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

# Ensure setup_logging is called only once if necessary
setup_logging()
logger = logging.getLogger(__name__)

class IndexProcessor:
    def __init__(self):
        self.config = load_config()
        self.repo_dir = os.path.join("indexing", "repos")
        os.makedirs(self.repo_dir, exist_ok=True)

        # --- ChromaDB Client Setup ---
        db_path = self.config.get("vector_db_path", os.path.join(project_root, "chroma_db"))
        logger.info(f"Initializing ChromaDB client at path: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)

        collection_name = self.config.get("collection_name", "code_embeddings")
        logger.info(f"Getting or creating collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # --- Model and Tokenizer Setup ---
        model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # --- Chunking Configuration ---
        self.max_chunk_length = self.config.get(
            "max_chunk_length",
            getattr(self.tokenizer, 'model_max_length', 512) # Use tokenizer's max length as a sensible default
        )
        # Ensure max_chunk_length doesn't exceed model's absolute max (if known)
        model_abs_max = getattr(self.embedding_model.config, 'max_position_embeddings', None)
        if model_abs_max and self.max_chunk_length > model_abs_max:
             logger.warning(f"Configured max_chunk_length ({self.max_chunk_length}) exceeds model's max_position_embeddings ({model_abs_max}). Clamping to {model_abs_max}.")
             self.max_chunk_length = model_abs_max

        self.stride = self.config.get("chunk_stride", int(self.max_chunk_length * 0.75))
        if self.stride >= self.max_chunk_length:
             logger.warning(f"Chunk stride ({self.stride}) is >= max_chunk_length ({self.max_chunk_length}). Setting stride to {int(self.max_chunk_length * 0.75)}.")
             self.stride = int(self.max_chunk_length * 0.75)

        logger.info(f"Chunking config: max_length={self.max_chunk_length}, stride={self.stride}")


    def index_code_files(self):
        programming_languages = self.config.get("programming_languages")
        if not programming_languages:
            logger.warning("No programming languages specified in config. Skipping indexing.")
            return

        logger.info(f"Starting indexing for languages: {programming_languages}")
        indexed_files = 0
        skipped_files = 0
        processed_chunks = 0
        error_files = 0

        for repo_name in os.listdir(self.repo_dir):
            repo_path = os.path.join(self.repo_dir, repo_name)
            if os.path.isdir(repo_path):
                logger.info(f"Processing repository: {repo_name}")
                try:
                    repo_stats = self._index_repo(repo_path, repo_name)
                    indexed_files += repo_stats['indexed']
                    skipped_files += repo_stats['skipped']
                    processed_chunks += repo_stats['chunks']
                    error_files += repo_stats['errors']
                except Exception as e:
                    logger.error(f"Unhandled exception while processing repo {repo_name}: {e}", exc_info=True)
                    # Depending on desired robustness, you might want to skip the whole repo or stop

        logger.info(f"Indexing complete. Indexed: {indexed_files} files, Skipped: {skipped_files} files, Errors: {error_files} files, Processed: {processed_chunks} chunks.")


    def _index_repo(self, repo_path, repo_name):
        repo_indexed_count = 0
        repo_skipped_count = 0
        repo_chunk_count = 0
        repo_error_count = 0
        for root, _, files in os.walk(repo_path):
            if ".git" in root.split(os.sep):
                continue
            for file in files:
                if not any(file.lower().endswith(lang.lower()) for lang in self.config["programming_languages"]):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_dir)
                file_id_base = relative_path.replace(os.sep, "_")

                first_chunk_id = f"{file_id_base}_chunk_0"
                try:
                    # Use limit=0 and count() for a potentially faster existence check
                    # existing_check = self.collection.get(ids=[first_chunk_id], limit=1, include=[]) # Faster if just checking existence
                    # Note: As of chromadb 0.4.x, get with limit=1 might be efficient enough. Count() can be slower sometimes. Stick with get for now.
                    existing_embeddings = self.collection.get(ids=[first_chunk_id], limit=1)
                    if existing_embeddings and existing_embeddings["ids"]:
                        logger.debug(f"Skipping already indexed file: {file_path}")
                        repo_skipped_count += 1
                        continue
                except Exception as e:
                     logger.error(f"Error checking existence for {first_chunk_id} in {file_path}: {e}")
                     repo_error_count += 1 # Count as error and skip
                     continue

                logger.info(f"Indexing file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()

                    if not code.strip():
                        logger.warning(f"Skipping empty file: {file_path}")
                        continue

                    # --- Generate embeddings ---
                    embeddings, chunk_input_ids_lists = self.generate_embeddings_for_chunks(code)
                    # --- End Generate embeddings ---

                    if not embeddings:
                         logger.warning(f"No embeddings generated for file: {file_path} (potentially due to errors in chunking/embedding)")
                         # Don't increment error count here if generate_embeddings handles its own errors
                         continue # Skip this file

                    ids_to_add = []
                    embeddings_to_add = []
                    metadatas_to_add = []
                    documents_to_add = []

                    for i, (embedding, chunk_ids_list) in enumerate(zip(embeddings, chunk_input_ids_lists)):
                        chunk_id = f"{file_id_base}_chunk_{i}"
                        chunk_metadata = {
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "repo_name": repo_name,
                            "chunk_index": i
                        }
                        chunk_text = self.tokenizer.decode(chunk_ids_list, skip_special_tokens=True)

                        ids_to_add.append(chunk_id)
                        embeddings_to_add.append(embedding)
                        metadatas_to_add.append(chunk_metadata)
                        documents_to_add.append(chunk_text)

                    if ids_to_add:
                         logger.debug(f"Adding {len(ids_to_add)} chunks for {file_path}")
                         self.collection.add(
                             embeddings=embeddings_to_add,
                             metadatas=metadatas_to_add,
                             documents=documents_to_add,
                             ids=ids_to_add
                         )
                         repo_indexed_count += 1
                         repo_chunk_count += len(ids_to_add)

                except FileNotFoundError:
                    logger.error(f"File not found during processing: {file_path}")
                    repo_error_count += 1
                except Exception as e:
                    # Catch errors during the file processing/embedding/adding phase
                    logger.error(f"Failed to process or index file {file_path}: {e}", exc_info=True)
                    repo_error_count += 1

        return {"indexed": repo_indexed_count, "skipped": repo_skipped_count, "chunks": repo_chunk_count, "errors": repo_error_count}


    def chunk_code_tokens(self, code):
        """Tokenizes code and returns overlapping chunks as lists of token IDs."""
        # IMPORTANT: Do NOT return tensors here. Return Python lists.
        inputs = self.tokenizer(
            code,
            max_length=self.max_chunk_length,
            stride=self.stride,
            truncation=True,
            return_overflowing_tokens=True,
            padding=False,  # No padding needed at this stage
            # return_tensors="pt"  <--- REMOVE THIS or set to None (default)
        )
        # inputs['input_ids'] is now a list of lists of integers
        return inputs['input_ids']

    def generate_embeddings_for_chunks(self, code):
        """Generates embeddings for overlapping code chunks."""
        # Get the tokenized chunks (list of lists of integers)
        chunked_input_ids_lists = self.chunk_code_tokens(code)

        all_embeddings = []
        # Store the original python lists of ids for potential decoding
        all_chunk_ids_lists = []

        self.embedding_model.eval()
        with torch.no_grad():
            for i, chunk_ids_list in enumerate(chunked_input_ids_lists):
                # logger.debug(f"Processing chunk {i+1}/{len(chunked_input_ids_lists)}, length: {len(chunk_ids_list)}")

                 # *** Convert the individual chunk (list of ints) to a tensor HERE ***
                try:
                    # Create tensor from the list
                    chunk_tensor = torch.tensor(chunk_ids_list)
                    # Add batch dimension and move to device
                    chunk_ids_batch = chunk_tensor.unsqueeze(0).to(self.device)

                    # Get model output
                    outputs = self.embedding_model(input_ids=chunk_ids_batch)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    all_embeddings.append(embedding.cpu().numpy().tolist())
                    # Store the original list of IDs
                    all_chunk_ids_lists.append(chunk_ids_list)

                except Exception as e:
                    logger.error(f"Error embedding chunk {i} (length {len(chunk_ids_list)}): {e}", exc_info=False) # Keep log concise
                    # Optionally skip this chunk and continue with the next
                    continue

        # Return the embeddings and the corresponding lists of token IDs
        return all_embeddings, all_chunk_ids_lists

# --- Rest of the file ---
if __name__ == "__main__":
    try:
        processor = IndexProcessor()
        processor.index_code_files()
    except Exception as main_err:
        logger.critical(f"An critical error occurred during IndexProcessor execution: {main_err}", exc_info=True)
        sys.exit(1)