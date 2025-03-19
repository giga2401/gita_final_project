import pandas as pd
from config import load_config

def run_evaluation():
    config = load_config()
    dataset = [
        {"code": "print('Hello, World!')", "is_plagiarized": False},
        {"code": "def add(a, b): return a + b", "is_plagiarized": True},
    ]

    results = []
    for case in dataset:
        # RAG Only
        rag_result = check_plagiarism(case["code"])
        # LLM Only (dummy implementation)
        llm_result = {"is_plagiarized": case["is_plagiarized"]}
        # Built System
        built_system_result = check_plagiarism(case["code"])

        results.append({
            "code": case["code"],
            "expected": case["is_plagiarized"],
            "rag_only": rag_result["is_plagiarized"],
            "llm_only": llm_result["is_plagiarized"],
            "built_system": built_system_result["is_plagiarized"]
        })

    # Save results as CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation results saved to evaluation_results.csv")

def check_plagiarism(code: str, config):
    import requests
    response = requests.post(
        f"http://localhost:{config['plagiarism_api_port']}/check_plagiarism",
        json={"code": code}
    )
    return response.json()

if __name__ == "__main__":
    run_evaluation()
