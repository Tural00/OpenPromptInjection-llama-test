# ollama_model.py
import requests
from OpenPromptInjection.models.Model import Model

class OllamaModel(Model):
    def __init__(self, config):
        self.model_name  = config["model_info"]["name"]          # e.g. "llama3:latest"
        self.base_url    = config["model_info"].get("base_url", "http://localhost:11434")
        self.temperature = config["params"].get("temperature", 0.1)
        self.max_tokens  = config["params"].get("max_output_tokens", 150)

    def print_model_info(self):
        print(f"[OllamaModel] model={self.model_name}  url={self.base_url}")

    def query(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()