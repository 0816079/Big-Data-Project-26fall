
import numpy as np
import os,json

def load_env(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

class Labeler:
    def __init__(self, kind, task, model="gpt-4o-mini", price=(0.15, 0.60), cache_dir="results/llm_cache"):
        self.kind, self.task, self.model, self.price = kind, task, model, price  
        self.calls = self.tokens_in = self.tokens_out = 0
        if kind == "llm":
            load_env()
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY not set")
            self.key = os.environ["OPENAI_API_KEY"]
            self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = f"{cache_dir}/{model.replace('/', '_')}.jsonl"
            self.cache = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file) as f:
                    self.cache = {r["key"]: r for r in map(json.loads, f)}