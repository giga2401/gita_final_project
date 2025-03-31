from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import chromadb
import logging
import time
import requests
import os
import sys
import json 
from typing import List, Optional 

# --- START: Python Path Modification (REMOVED) ---

from shared.config import load_config, get_openai_api_key
from shared.utils import setup_logging
try:
    from .prompt import PLAGIARISM_CHECK_PROMPT
except ImportError:
    from prompt import PLAGIARISM_CHECK_PROMPT

from openai import OpenAI

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Determines project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    project_root = os.path.abspath('.')

IS_DOCKER_ENV = os.environ.get('IS_DOCKER_ENV', 'false').lower() == 'true'
logger.info(f"Running in Docker environment: {IS_DOCKER_ENV}")

# --- Global Variables & Initialization ---
try:
    config = load_config()
    openai_client = OpenAI(api_key=get_openai_api_key())

    # --- Gets Embedding Service URL ---
    default_embedding_host = "embedding_server" if IS_DOCKER_ENV else "localhost"
    embedding_port = config.get("embedding_api_port", 8001)
    EMBEDDING_URL = os.environ.get("EMBEDDING_SERVICE_URL", f"http://{default_embedding_host}:{embedding_port}") + "/embed"
    logger.info(f"Using Embedding Service at: {EMBEDDING_URL}")

    # --- ChromaDB Client ---
    db_path_from_config = config["vector_db_path"]
    if not os.path.isabs(db_path_from_config):
         db_path = os.path.join(project_root, db_path_from_config)
    else:
         db_path = db_path_from_config
    logger.info(f"Connecting to ChromaDB at resolved path: {db_path}")
    db_parent_dir = os.path.dirname(db_path)
    if db_parent_dir: os.makedirs(db_parent_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path)

    collection_name = config["collection_name"]
    try:
        chroma_client.heartbeat()
        logger.info("ChromaDB connection successful.")
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Connected to existing collection: {collection_name} (Count: {collection.count()})")
    except Exception as e:
        logger.error(f"FATAL: Failed to get ChromaDB collection '{collection_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    # --- Configurable Parameters ---
    RAG_NUM_RESULTS = config.get("rag_num_results", 10)
    MAX_LLM_CONTEXT_LENGTH = config.get("max_llm_context_length", 10000)
    SIMILARITY_THRESHOLD_EARLY_EXIT = config.get("plagiarism_similarity_threshold")
    if SIMILARITY_THRESHOLD_EARLY_EXIT is not None:
        try:
            SIMILARITY_THRESHOLD_EARLY_EXIT = float(SIMILARITY_THRESHOLD_EARLY_EXIT)
            logger.info(f"Using early exit similarity threshold (L2^2): {SIMILARITY_THRESHOLD_EARLY_EXIT:.4f}")
        except ValueError:
            logger.error(f"Invalid plagiarism_similarity_threshold value: {SIMILARITY_THRESHOLD_EARLY_EXIT}. Disabling threshold.")
            SIMILARITY_THRESHOLD_EARLY_EXIT = None
    else:
        logger.info("No early exit similarity threshold configured.")


except (RuntimeError, ValueError, Exception) as E:
     logger.critical(f"Failed to initialize application dependencies: {E}", exc_info=True)
     sys.exit(1)

# --- FastAPI App ---
app = FastAPI()

# --- Middleware for Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    log_extra = {"client_host": request.client.host} if request.client else {}
    logger.info(f"Request received: {request.method} {request.url.path}", extra=log_extra)
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        log_extra["status_code"] = response.status_code
        log_extra["process_time_ms"] = int(process_time * 1000)
        logger.info(f"Request completed: {request.method} {request.url.path}", extra=log_extra)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        log_extra["process_time_ms"] = int(process_time * 1000)
        if isinstance(e, HTTPException) and e.status_code < 500:
             logger.warning(f"Request failed (Client Error): {request.method} {request.url.path} - Status: {e.status_code} Detail: {e.detail}", extra=log_extra)
        else:
             logger.error(f"Request failed (Server Error): {request.method} {request.url.path} - Error: {e}", exc_info=True, extra=log_extra)
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail="Internal Server Error") from e
        raise e


# --- Pydantic Model ---
class CodeSnippet(BaseModel):
    code: str

# --- Helper: Get Embedding via Service ---
def get_embedding_from_service(code: str) -> Optional[List[float]]:
    """Calls the embedding service to get an embedding for the code."""
    try:
        response = requests.post(EMBEDDING_URL, json={"text": code}, timeout=120)
        response.raise_for_status()
        data = response.json()
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        else:
            logger.error(f"Invalid embedding format received from service: {data}")
            return None
    except requests.exceptions.Timeout:
         logger.error(f"Embedding service request timed out after 120s.")
         raise HTTPException(status_code=504, detail="Embedding service timed out after 120s.")
    except requests.exceptions.RequestException as req_err:
         status = req_err.response.status_code if req_err.response else "N/A"
         text = req_err.response.text[:200] if req_err.response else "N/A"
         logger.error(f"Embedding service request failed: {req_err} (Status: {status}, Response: {text})", exc_info=True)
         raise HTTPException(status_code=502, detail=f"Embedding service error (Status: {status})")
    except Exception as emb_svc_err:
         logger.error(f"Unexpected error contacting embedding service: {emb_svc_err}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to get embedding from service.")

# --- API Endpoint ---
@app.post("/check_plagiarism")
async def check_plagiarism(snippet: CodeSnippet):
    if not snippet.code or not snippet.code.strip():
        logger.warning("Received request with empty code snippet.")
        raise HTTPException(status_code=400, detail="Code snippet cannot be empty.")

    request_start_time = time.time()
    logger.info(f"Checking plagiarism for code snippet (length: {len(snippet.code)})...")

    try:
        # 1. Gets Embedding for Input Code
        logger.debug("Getting embedding for input code...")
        embedding_start_time = time.time()
        embedding = get_embedding_from_service(snippet.code) # Uses updated timeout
        if not embedding:
             raise HTTPException(status_code=500, detail="Failed to retrieve embedding.")
        embedding_time = time.time() - embedding_start_time
        logger.debug(f"Embedding received ({embedding_time:.4f}s).")

        # 2. Queries Vector Database
        logger.debug(f"Querying vector database for {RAG_NUM_RESULTS} results...")
        db_query_start_time = time.time()
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=RAG_NUM_RESULTS,
                include=["metadatas", "documents", "distances"]
            )
            db_query_time = time.time() - db_query_start_time
            logger.debug(f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results ({db_query_time:.4f}s).")
        except Exception as db_err:
            logger.error(f"ChromaDB query failed: {db_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database query failed: {db_err}")

        # 3. Prepares Context & Checks Threshold
        similar_docs = results.get("documents", [[]])[0]
        similar_metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not similar_docs:
             logger.info("No similar documents found in the database. Likely not plagiarized.")
             return {"is_plagiarized": False, "reasoning": "No similar code found in the database.", "references": []}

        # --- Early Exit based on Distance ---
        if SIMILARITY_THRESHOLD_EARLY_EXIT is not None and distances:
            min_distance = distances[0]
            logger.debug(f"Closest match distance (L2^2): {min_distance:.4f}")
            if min_distance > SIMILARITY_THRESHOLD_EARLY_EXIT:
                logger.info(f"Minimum distance {min_distance:.4f} exceeds threshold {SIMILARITY_THRESHOLD_EARLY_EXIT:.4f}. Concluding 'Not Plagiarized' early.")
                return {
                    "is_plagiarized": False,
                    "reasoning": f"Code similarity below threshold (Closest match distance: {min_distance:.4f}).",
                    "references": []
                }

        context_parts = []
        unique_refs = {}
        for i, (doc, meta, dist) in enumerate(zip(similar_docs, similar_metadatas, distances)):
            if not meta: meta = {}
            file_path = meta.get("relative_path", "Unknown path")
            repo_name = meta.get("repo_name", "Unknown repo")
            chunk_idx = meta.get("chunk_index", "N/A")
            ref_key = f"{repo_name}__{file_path}"
            if ref_key not in unique_refs:
                unique_refs[ref_key] = {"repo": repo_name, "file": file_path, "closest_chunk": chunk_idx, "distance": dist}
            context_parts.append(f"--- Match {i+1} (Distance: {dist:.4f}) ---\nRepo: {repo_name}\nFile: {file_path}\nChunk Index: {chunk_idx}\nCode Snippet:\n```\n{doc}\n```\n")

        context = "\n".join(context_parts)

        # Truncates context if too long
        if len(context) > MAX_LLM_CONTEXT_LENGTH * 1.5:
            logger.warning(f"Context length ({len(context)} chars) significantly exceeds rough limit based on {MAX_LLM_CONTEXT_LENGTH}. Truncating.")
            context = context[:int(MAX_LLM_CONTEXT_LENGTH * 1.5)] + "\n... (context truncated due to length)"

        # 4. Calls LLM for Plagiarism Decision + Reasoning (JSON Output)
        logger.debug("Calling LLM for plagiarism decision + reasoning (JSON output)...")
        llm_start_time = time.time()
        prompt = PLAGIARISM_CHECK_PROMPT.format(
            user_code=snippet.code,
            context=context
        )

        try:
            response = openai_client.chat.completions.create(
                model=config["llm_model"],
                messages=[
                    {"role": "system", "content": "You are a code plagiarism detection assistant. Respond ONLY with a valid JSON object containing keys 'is_plagiarized' (boolean) and 'reasoning' (string, 1-3 sentences) based on the user prompt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150 
            )
            llm_time = time.time() - llm_start_time
            logger.debug(f"LLM call completed ({llm_time:.4f}s).")

            llm_response_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM Raw JSON Output: '{llm_response_text}'")

            # --- PARSES JSON ---
            is_plagiarized = False # Default
            reasoning = "Failed to parse LLM JSON response." # Default
            try:
                if llm_response_text.startswith("```json"):
                    llm_response_text = llm_response_text.replace("```json", "").replace("```", "").strip()
                decision_data = json.loads(llm_response_text)

                # Safely gets 'is_plagiarized'
                plag_value = decision_data.get("is_plagiarized")
                if isinstance(plag_value, bool):
                    is_plagiarized = plag_value
                else:
                    logger.error(f"LLM JSON response missing or invalid 'is_plagiarized' boolean key: {llm_response_text}")

                # Safely gets 'reasoning'
                reasoning_value = decision_data.get("reasoning")
                if isinstance(reasoning_value, str):
                    reasoning = reasoning_value
                else:
                     logger.error(f"LLM JSON response missing or invalid 'reasoning' string key: {llm_response_text}")
                     reasoning = "Failed to parse reasoning from LLM." 

                logger.info(f"Parsed LLM JSON decision: {is_plagiarized}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode LLM JSON response: '{llm_response_text}'. Defaulting decision to 'No'.")
                is_plagiarized = False # Default on JSON error
                reasoning = "LLM output was not valid JSON."

        except Exception as llm_err:
            logger.error(f"OpenAI API call failed: {llm_err}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"LLM service error: {llm_err}")

        # 5. Formats and Returns Response
        logger.info(f"Plagiarism decision: {'Yes' if is_plagiarized else 'No'}. Reasoning: {reasoning}")
        references = list(unique_refs.values()) if is_plagiarized else []
        total_request_time = time.time() - request_start_time
        logger.info(f"Total request processing time: {total_request_time:.4f}s")

        return {
            "is_plagiarized": is_plagiarized,
            "reasoning": reasoning,
            "references": references
        }

    except HTTPException:
        raise # Re-raises known HTTP exceptions
    except Exception as e:
        # any other unexpected errors during the request handling
        logger.error(f"Unexpected error in check_plagiarism: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# Health check endpoint
@app.get("/")
def read_root():
    try:
        chroma_client.heartbeat()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"
        logger.error("Health check failed: DB connection error.")

    return {"status": "Plagiarism Checker API is running", "database_status": db_status}

# --- Uvicorn Entry Point (for local run) ---
if __name__ == "__main__":
    import uvicorn
    api_port = config.get("plagiarism_api_port", 8000)
    logger.info(f"Starting Uvicorn server locally on port {api_port} with reload enabled.")
    uvicorn.run("plag_checker.main:app", host="0.0.0.0", port=api_port, reload=True)