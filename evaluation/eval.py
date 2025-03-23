import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from shared.config import load_config
import requests
import logging
from shared.utils import setup_logging

# Set up logging
setup_logging()

def run_evaluation():
    """
    Main function to run the evaluation process.
    """
    logging.info("Starting evaluation process...")

    # Load configuration
    try:
        config = load_config()
        logging.info("Configuration loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return

    # Load dataset
    dataset_path = config.get("dataset_path", "data/dataset.csv")
    logging.info(f"Attempting to load dataset from: {dataset_path}")

    try:
        dataset = pd.read_csv(dataset_path)
        logging.info(f"Dataset loaded successfully with {len(dataset)} rows.")
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {dataset_path}. Please ensure the file exists.")
        return
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # Check if the required columns exist
    if "SNIPPET" not in dataset.columns or "PLAGIARIZED" not in dataset.columns:
        logging.error("Dataset must contain 'SNIPPET' and 'PLAGIARIZED' columns.")
        return

    # Initialize results list
    results = []

    # Process each row in the dataset
    for index, row in dataset.iterrows():
        try:
            logging.info(f"Processing row {index + 1}/{len(dataset)}: {row['SNIPPET'][:50]}...")  # Log first 50 chars of snippet

            # Check plagiarism using RAG-only, LLM-only, and Built System
            rag_result = check_plagiarism(row["SNIPPET"], config, method="rag")
            llm_result = check_plagiarism(row["SNIPPET"], config, method="llm")
            built_system_result = check_plagiarism(row["SNIPPET"], config, method="built_system")

            # Append results
            results.append({
                "code": row["SNIPPET"],
                "expected": row["PLAGIARIZED"],
                "rag_only": rag_result["is_plagiarized"],
                "llm_only": llm_result["is_plagiarized"],
                "built_system": built_system_result["is_plagiarized"]
            })
        except Exception as e:
            logging.error(f"Error processing row {index + 1}: {e}")
            continue

    # Calculate evaluation metrics for each method
    metrics = {
        "rag_only": calculate_metrics(results, "rag_only"),
        "llm_only": calculate_metrics(results, "llm_only"),
        "built_system": calculate_metrics(results, "built_system")
    }

    # Log detailed metrics
    for method, method_metrics in metrics.items():
        logging.info(f"Metrics for {method}:")
        logging.info(f"  Precision: {method_metrics['precision']:.4f}")
        logging.info(f"  Recall: {method_metrics['recall']:.4f}")
        logging.info(f"  F1 Score: {method_metrics['f1_score']:.4f}")

    # Save results to a CSV file
    try:
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results.csv", index=False)
        logging.info("Evaluation results saved to 'evaluation_results.csv'.")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")

def calculate_metrics(results, method):
    """
    Calculate precision, recall, and F1 score for a specific method.

    Args:
        results (list): List of dictionaries containing evaluation results.
        method (str): The method to calculate metrics for (e.g., "rag_only", "llm_only", "built_system").

    Returns:
        dict: Dictionary containing precision, recall, and F1 score.
    """
    if not results:
        logging.error("No results to calculate metrics.")
        return {}

    # Extract true and predicted labels
    y_true = [result["expected"] for result in results]
    y_pred = [result[method] for result in results]

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def check_plagiarism(code: str, config, method="built_system"):
    """
    Check if the given code snippet is plagiarized using the specified method.

    Args:
        code (str): The code snippet to check.
        config (dict): Configuration dictionary.
        method (str): The method to use ("rag_only", "llm_only", or "built_system").

    Returns:
        dict: Dictionary containing the plagiarism check result.
    """
    try:
        logging.info(f"Checking plagiarism using {method}...")
        if method == "rag_only":
            # Simulate RAG-only result (replace with actual implementation)
            return {"is_plagiarized": False}
        elif method == "llm_only":
            # Simulate LLM-only result (replace with actual implementation)
            return {"is_plagiarized": False}
        else:
            # Use the built system
            response = requests.post(
                f"http://localhost:{config['plagiarism_api_port']}/check_plagiarism",
                json={"code": code},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking plagiarism using {method}: {e}")
        return {"is_plagiarized": False}

if __name__ == "__main__":
    run_evaluation()