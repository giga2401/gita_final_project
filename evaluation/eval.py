import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import requests
import logging
import os
import sys
import time
import json
import csv 
from typing import Optional
from shared.config import load_config, get_openai_api_key
from shared.utils import setup_logging

import chromadb
from openai import OpenAI

setup_logging()
logger = logging.getLogger(__name__)

# Determines project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    project_root = os.path.abspath('.')
logger.info(f"Project root determined as: {project_root}")

IS_DOCKER_ENV = os.environ.get('IS_DOCKER_ENV', 'false').lower() == 'true'
logger.info(f"Running in Docker environment: {IS_DOCKER_ENV}")

# --- Global Config & Clients ---
try:
    config = load_config()

    # --- API URLs ---
    default_api_host = "plagiarism_checker" if IS_DOCKER_ENV else "localhost"
    api_host = os.environ.get("API_HOST", default_api_host)
    api_port = config['plagiarism_api_port']
    rag_llm_api_url = f"http://{api_host}:{api_port}/check_plagiarism"
    health_check_url = f"http://{api_host}:{api_port}/" # Health check

    default_embedding_host = "embedding_server" if IS_DOCKER_ENV else "localhost"
    embedding_port = config.get("embedding_api_port", 8001)
    embedding_service_url = os.environ.get("EMBEDDING_SERVICE_URL", f"http://{default_embedding_host}:{embedding_port}") + "/embed" # Use single embed endpoint

    logger.info(f"Using Plagiarism API URL: {rag_llm_api_url}")
    logger.info(f"Using Embedding Service URL: {embedding_service_url}")

    # Clients
    openai_client = OpenAI(api_key=get_openai_api_key())
    llm_only_model = config["llm_model"] # Uses the same upgraded model

    db_path_from_config = config["vector_db_path"]
    if not os.path.isabs(db_path_from_config): db_path = os.path.join(project_root, db_path_from_config)
    else: db_path = db_path_from_config
    logger.info(f"Connecting to ChromaDB for RAG-only eval at: {db_path}")
    db_parent_dir = os.path.dirname(db_path); os.makedirs(db_parent_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection_name = config["collection_name"]
    try:
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"ChromaDB connection successful for RAG-only eval. Collection count: {collection.count()}")
    except Exception as chroma_err:
        logger.error(f"Failed to connect to Chroma collection '{collection_name}' for RAG-only eval: {chroma_err}. RAG-only eval will fail.")
        collection = None # Ensures RAG-only fails gracefully

    # --- Thresholds ---
    # Distance threshold (e.g., L2 squared). Lower = more similar.
    RAG_ONLY_DISTANCE_THRESHOLD = float(config.get("rag_only_threshold", 0.8))
    logger.info(f"Using RAG-only distance threshold (L2^2): {RAG_ONLY_DISTANCE_THRESHOLD}")

except (RuntimeError, ValueError, KeyError, Exception) as init_err:
    logger.critical(f"CRITICAL: Failed to load configuration or initialize clients: {init_err}", exc_info=True)
    sys.exit(1)

# --- LLM-Only Prompt Template ---
LLM_ONLY_PROMPT_TEMPLATE = """
You are a highly specialized AI assistant for detecting code plagiarism.  
Your task is to determine if the provided code snippet is likely plagiarized based *only* on its internal analysis, without external references.  

### **Code to Analyze:**  
{code}  

### **Instructions:**  
1. **Evaluate Originality:** Analyze the **logic, structure, algorithms, implementation details, comments, and variable naming patterns** within the snippet.  
2. **Consider Common Patterns vs. Unique Implementations:**  
   - **Common Patterns:** Standard library usage, conventional syntax, basic programming constructs.  
   - **Unique Implementations:** Distinctive structuring, uncommon logic, or patterns suggesting non-originality.  
3. **Determine Likelihood of Plagiarism:** Assess whether the code appears to be an original implementation or potentially copied.  
4. **Output JSON-ONLY:** Respond strictly in JSON format with:  
   - `"is_plagiarized"`: **boolean** (`true` / `false`)  

#### **Example Output:**  
```json
{{
    "is_plagiarized": false
}}

Your JSON Response:
"""


# --- Helper: Gets Embedding (for RAG-only) ---
def get_embedding_from_service(code: str) -> Optional[list[float]]:
    if not code or not code.strip(): return None
    try:
        response = requests.post(embedding_service_url, json={"text": code}, timeout=120)
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.Timeout: # Specific timeout handling
        logger.error(f"Failed to get embedding from service: Request timed out after {120}s")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get embedding from service: {e}")
    except Exception as e:
        logger.error(f"Unexpected error getting embedding: {e}")
    return None

# --- Helper: Parses LLM Yes/No Response
def parse_yes_no(response_text: str) -> int:
    """Parses 'Plagiarism: Yes' or 'Plagiarism: No', returns 1 for Yes, 0 for No, -1 for error."""
    if not response_text: return -1
    lines = response_text.strip().lower().splitlines()
    for line in reversed(lines):
        if line.startswith("plagiarism:"):
            decision = line.replace("plagiarism:", "").strip()
            if decision == "yes": return 1
            if decision == "no": return 0
    logger.warning(f"Could not parse 'Plagiarism: Yes/No' from: '{response_text}'")
    return -1 # Error/Unknown

def run_evaluation():
    logger.info("Starting evaluation process...")

    # --- Wait for API Server ---
    max_wait = 90; wait_interval = 5; waited = 0; api_ready = False
    logger.info(f"Waiting up to {max_wait}s for plagiarism API at {health_check_url}...")
    while waited < max_wait:
         try:
             # Use a timeout on the health check itself
             response = requests.get(health_check_url, timeout=wait_interval - 1)
             if response.status_code == 200:
                 logger.info("Plagiarism API health check successful.")
                 api_ready = True
                 break
             else:
                 logger.warning(f"Health check attempt failed: Status {response.status_code}")
         except requests.exceptions.RequestException as e:
             logger.warning(f"Health check attempt failed: {e}")
         time.sleep(wait_interval)
         waited += wait_interval
         if waited < max_wait: logger.info(f"Still waiting for API... ({waited}s / {max_wait}s)")

    if not api_ready:
        logger.error(f"Plagiarism API did not become ready within {max_wait} seconds. Aborting.")
        return

    # --- Load Dataset ---
    dataset_path_config = config.get("dataset_path", "./data/dataset.csv") # Default relative to project
    if not os.path.isabs(dataset_path_config):
        dataset_path = os.path.join(project_root, dataset_path_config.lstrip('./').lstrip('/'))
    else:
        dataset_path = dataset_path_config
    logger.info(f"Attempting to load dataset from: {dataset_path}")
    try:
        if not os.path.exists(dataset_path):
             raise FileNotFoundError(f"Dataset file not found at resolved path: {dataset_path}")
        dataset = pd.read_csv(dataset_path)
        if "SNIPPET" not in dataset.columns or "PLAGIARIZED" not in dataset.columns:
             raise ValueError("Dataset must contain 'SNIPPET' and 'PLAGIARIZED' columns.")
        # Ensure correct types
        dataset['SNIPPET'] = dataset['SNIPPET'].astype(str)
        dataset['PLAGIARIZED'] = dataset['PLAGIARIZED'].astype(int)
        logger.info(f"Dataset loaded successfully with {len(dataset)} rows.")
    except Exception as e:
        logger.error(f"Error loading or processing dataset from {dataset_path}: {e}", exc_info=True)
        return

    all_results = []
    request_timeout = 600 # Longer timeout for potentially slow RAG+LLM calls

    logger.info(f"Processing {len(dataset)} dataset rows for all 3 approaches...")
    for index, row in dataset.iterrows():
        code_snippet = row["SNIPPET"]
        expected_label = row["PLAGIARIZED"]
        row_identifier = f"Row {index + 1}/{len(dataset)}"

        if not code_snippet or not code_snippet.strip():
             logger.warning(f"{row_identifier}: Skipping empty code snippet.")
             all_results.append({ "row_index": index, "snippet_preview": "EMPTY", "expected": expected_label,
                                 "predicted_rag_llm": -1, "predicted_llm_only": -1, "predicted_rag_only": -1,
                                 "rag_llm_reasoning": "Skipped", "rag_llm_references": "[]", "rag_only_distance": None })
             continue

        logger.debug(f"Processing {row_identifier}...")
        # Initialize results for this row
        result_row = {
            "row_index": index,
            "snippet_preview": code_snippet[:80].replace('\n', '\\n') + "...",
            "expected": expected_label,
            "predicted_rag_llm": -1,
            "predicted_llm_only": -1,
            "predicted_rag_only": -1,
            "rag_llm_reasoning": None,
            "rag_llm_references": "[]",
            "rag_only_distance": None
        }

        # --- 1. RAG + LLM (Your System via API) ---
        try:
            logger.debug(f"{row_identifier} - Calling RAG+LLM API...")
            response = requests.post(rag_llm_api_url, json={"code": code_snippet}, timeout=request_timeout) # Use updated timeout
            response.raise_for_status()
            api_result = response.json()
            logger.info(f"{row_identifier} (RAG+LLM) RAW Response: {json.dumps(api_result)}") # Log the raw JSON result

            if "is_plagiarized" in api_result and isinstance(api_result["is_plagiarized"], bool):
                is_plagiarized_bool = api_result["is_plagiarized"]
                result_row["predicted_rag_llm"] = 1 if is_plagiarized_bool else 0
                # Store the placeholder reasoning or the bool value itself if you prefer
                result_row["rag_llm_reasoning"] = api_result.get("reasoning", "N/A")
                logger.debug(f"{row_identifier} (RAG+LLM): Parsed Pred Bool={is_plagiarized_bool}, Int={result_row['predicted_rag_llm']}")
            else:
                logger.error(f"{row_identifier} (RAG+LLM): API response missing 'is_plagiarized' boolean key: {json.dumps(api_result)}")
                result_row["predicted_rag_llm"] = -1 # Mark as error
                result_row["rag_llm_reasoning"] = "Invalid API Response Format"
            # --- End With ---

            references = api_result.get("references", [])
            result_row["rag_llm_references"] = json.dumps(references) if references else "[]" # Store as JSON string 

        except requests.exceptions.Timeout:
            logger.error(f"{row_identifier} (RAG+LLM): Request timed out ({request_timeout}s).")
            result_row["rag_llm_reasoning"] = "Timeout Error"
        except requests.exceptions.RequestException as e:
            err_detail = f"API Error: Status {e.response.status_code}" if e.response else f"API Error: {e}"
            logger.error(f"{row_identifier} (RAG+LLM): {err_detail}")
            result_row["rag_llm_reasoning"] = err_detail
        except Exception as e:
            logger.error(f"{row_identifier} (RAG+LLM): Unexpected error: {e}", exc_info=True)
            result_row["rag_llm_reasoning"] = f"Unexpected Error: {e}"

        # --- 2. LLM Only ---
        try:
            logger.debug(f"{row_identifier} - Calling LLM Only (JSON output)...")
            llm_response = openai_client.chat.completions.create(
                model=llm_only_model,
                messages=[
                    {"role": "system", "content": "Respond ONLY with a valid JSON object containing the key 'is_plagiarized' (boolean value)."}, # System prompt for LLM-Only
                    {"role": "user", "content": LLM_ONLY_PROMPT_TEMPLATE.format(code=code_snippet)}
                ],
                temperature=0.7,
                max_tokens=20 # Allows room for JSON
            )
            llm_text = llm_response.choices[0].message.content.strip()
            logger.debug(f"{row_identifier} (LLM Only) RAW JSON Output: '{llm_text}'")

            # --- PARSE JSON for LLM_ONLY ---
            predicted_llm_only_bool = False # Default
            try:
                if llm_text.startswith("```json"):
                    llm_text = llm_text.replace("```json", "").replace("```", "").strip()
                decision_data = json.loads(llm_text)
                if isinstance(decision_data.get("is_plagiarized"), bool):
                    predicted_llm_only_bool = decision_data["is_plagiarized"]
                    result_row["predicted_llm_only"] = 1 if predicted_llm_only_bool else 0
                else:
                    logger.error(f"{row_identifier} (LLM Only) JSON missing 'is_plagiarized' boolean key: {llm_text}")
                    result_row["predicted_llm_only"] = -1 # Mark as error
            except json.JSONDecodeError:
                logger.error(f"{row_identifier} (LLM Only) Failed to decode JSON: '{llm_text}'.")
                result_row["predicted_llm_only"] = -1 # Mark as error
            # --- END JSON PARSING ---

            logger.debug(f"{row_identifier} (LLM Only): Parsed Pred Bool={predicted_llm_only_bool}, Int={result_row['predicted_llm_only']}")

        except Exception as e:
            logger.error(f"{row_identifier} (LLM Only): OpenAI API call failed: {e}", exc_info=True)
            result_row["predicted_llm_only"] = -1

        # --- 3. RAG Only ---
        if collection: # Only run if Chroma connection was successful
            try:
                logger.debug(f"{row_identifier} - Performing RAG Only check...")
                embedding = get_embedding_from_service(code_snippet)
                if embedding:
                    query_results = collection.query(
                        query_embeddings=[embedding], n_results=1, include=["distances"]
                    )
                    distances = query_results.get('distances', [[]])[0]
                    if distances:
                        rag_dist = distances[0] # L2 squared distance
                        result_row["rag_only_distance"] = f"{rag_dist:.4f}"
                        # Lower distance means more similar -> Plagiarized
                        result_row["predicted_rag_only"] = 1 if rag_dist <= RAG_ONLY_DISTANCE_THRESHOLD else 0
                        logger.debug(f"{row_identifier} (RAG Only): Dist^2={rag_dist:.4f} (Thresh={RAG_ONLY_DISTANCE_THRESHOLD}) -> Pred={result_row['predicted_rag_only']}")
                    else:
                        logger.warning(f"{row_identifier} (RAG Only): No results from ChromaDB.")
                        result_row["predicted_rag_only"] = 0 # Treat as not plagiarized if nothing found
                else:
                    logger.error(f"{row_identifier} (RAG Only): Failed to get embedding.")
            except Exception as e:
                 logger.error(f"{row_identifier} (RAG Only): Error during query or processing: {e}", exc_info=True)
        else:
            logger.warning(f"{row_identifier} (RAG Only): Skipping due to ChromaDB connection failure.")

        all_results.append(result_row)
    # --- End Process Dataset Rows ---

    if not all_results:
        logger.warning("No results were generated. Cannot calculate metrics or save files.")
        return

    results_df = pd.DataFrame(all_results)

    # --- Output Directory (relative to project root) ---
    output_dir = os.path.join(project_root, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, "evaluation_results_comparison.csv")
    metrics_csv_path = os.path.join(output_dir, "evaluation_metrics.csv")

    # --- Saving Raw Results CSV ---
    try:
        # Select and order columns for clarity
        results_cols = [ "row_index", "snippet_preview", "expected",
                         "predicted_rag_llm", "predicted_llm_only", "predicted_rag_only",
                         "rag_llm_reasoning", "rag_llm_references", "rag_only_distance" ]
        results_df[results_cols].to_csv(results_csv_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Raw evaluation results saved to '{results_csv_path}'.")
    except Exception as e:
        logger.error(f"Error saving raw results CSV: {e}", exc_info=True)

    # --- Calculates and Saves Metrics CSV ---
    logger.info("--- Calculating Evaluation Metrics ---")
    metrics_data = []
    approaches = ["rag_llm", "llm_only", "rag_only"]

    for approach in approaches:
        pred_col = f"predicted_{approach}"
        # Filters out rows where prediction failed (-1)
        valid_df = results_df[results_df[pred_col].isin([0, 1])].copy()

        if valid_df.empty:
            logger.warning(f"No valid results (0 or 1) for approach '{approach}'. Skipping metrics calculation.")
            metrics_data.append({ "approach": approach, "error": "No valid predictions" })
            continue

        y_true = valid_df["expected"]
        y_pred = valid_df[pred_col]
        total_evaluated = len(valid_df)

        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) 
            tn, fp, fn, tp = cm.ravel()

            metrics = {
                "approach": approach,
                "total_evaluated": total_evaluated,
                "accuracy": f"{accuracy:.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1_score": f"{f1:.4f}",
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                 # Stores CM as string/list for CSV compatibility
                "confusion_matrix (TN,FP;FN,TP)": str(cm.tolist()),
                "error": None # No error
            }
            metrics_data.append(metrics)

            logger.info(f"\n--- Metrics for Approach: {approach.upper()} ---")
            logger.info(f"  Total Valid Rows Evaluated: {metrics['total_evaluated']}")
            logger.info(f"  Accuracy:  {metrics['accuracy']}")
            logger.info(f"  Precision: {metrics['precision']}")
            logger.info(f"  Recall:    {metrics['recall']}")
            logger.info(f"  F1 Score:  {metrics['f1_score']}")
            logger.info(f"  Confusion Matrix:\n{cm}")
            logger.info("--------------------------")

        except Exception as e:
             logger.error(f"Error calculating metrics for approach '{approach}': {e}", exc_info=True)
             metrics_data.append({ "approach": approach, "error": str(e) })

    # --- Saves Metrics Summary CSV ---
    if metrics_data:
        try:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_cols = ["approach", "total_evaluated", "accuracy", "precision", "recall", "f1_score",
                            "true_negatives", "false_positives", "false_negatives", "true_positives",
                            "confusion_matrix (TN,FP;FN,TP)", "error"]
            # Ensures all expected columns exist, adding missing ones with None/NaN
            for col in metrics_cols:
                if col not in metrics_df.columns:
                    metrics_df[col] = None
            metrics_df[metrics_cols].to_csv(metrics_csv_path, index=False)
            logger.info(f"Evaluation metrics summary saved to '{metrics_csv_path}'.")
        except Exception as e:
            logger.error(f"Error saving evaluation metrics CSV: {e}", exc_info=True)
    else:
        logger.warning("No metrics data was generated to save.")

if __name__ == "__main__":
    run_evaluation()
    logger.info("Evaluation script finished.")