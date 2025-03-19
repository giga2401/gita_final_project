import os
import openai
from config import get_openai_api_key

class CodeChunker:
    def __init__(self, model_name="gpt-3.5-turbo"):  # Updated model to gpt-3.5-turbo
        """Initializes OpenAI model."""
        self.api_key = get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key is missing.")
        
        openai.api_key = self.api_key
        self.model_name = model_name

    def process_repository(self, repo_path):
        """Processes repository files and returns code chunks."""
        chunks = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".py", ".js", ".java", ".cpp")):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        chunks.append(self.chunk_code(code, file_path))
        return chunks

    def chunk_code(self, code, file_path):
        """Chunks the code and summarizes it using OpenAI."""
        client = openai.OpenAI(api_key=self.api_key)  # Use the new OpenAI client
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Summarize this code snippet."},
                {"role": "user", "content": code},
            ]
        )
        summary = response.choices[0].message.content
        return {"file": file_path, "summary": summary}
