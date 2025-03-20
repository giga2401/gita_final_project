import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from config import load_config
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_evaluation():
    config = load_config()
    dataset = [
        {"code": "print('Hello, World!')", "is_plagiarized": False},
        {"code": "def add(a, b): return a + b", "is_plagiarized": True},
        # Add more test cases here
    ]

    results = []
    for case in dataset:
        # RAG Only
        rag_result = check_plagiarism(case["code"], config)
        # LLM Only (dummy implementation)
        llm_result = {"is_plagiarized": case["is_plagiarized"]}
        # Built System
        built_system_result = check_plagiarism(case["code"], config)

        results.append({
            "code": case["code"],
            "expected": case["is_plagiarized"],
            "rag_only": rag_result["is_plagiarized"],
            "llm_only": llm_result["is_plagiarized"],
            "built_system": built_system_result["is_plagiarized"]
        })

    # Calculate metrics
    metrics = calculate_metrics(results)
    logging.info(f"Evaluation metrics: {metrics}")

    # Save results as CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    logging.info("Evaluation results saved to evaluation_results.csv")

def calculate_metrics(results):
    y_true = [case["expected"] for case in results]
    y_pred_rag = [case["rag_only"] for case in results]
    y_pred_llm = [case["llm_only"] for case in results]
    y_pred_system = [case["built_system"] for case in results]

    metrics = {
        "rag": {
            "precision": precision_score(y_true, y_pred_rag),
            "recall": recall_score(y_true, y_pred_rag),
            "f1": f1_score(y_true, y_pred_rag)
        },
        "llm": {
            "precision": precision_score(y_true, y_pred_llm),
            "recall": recall_score(y_true, y_pred_llm),
            "f1": f1_score(y_true, y_pred_llm)
        },
        "system": {
            "precision": precision_score(y_true, y_pred_system),
            "recall": recall_score(y_true, y_pred_system),
            "f1": f1_score(y_true, y_pred_system)
        }
    }
    return metrics

def check_plagiarism(code: str, config):
    try:
        response = requests.post(
            f"http://localhost:{config['plagiarism_api_port']}/check_plagiarism",
            json={"code": code}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking plagiarism: {e}")
        return {"is_plagiarized": False}

if __name__ == "__main__":
    run_evaluation()