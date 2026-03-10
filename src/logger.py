import time
import json
import logging
import os
import random
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv
import ollama

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()                       # reads .env file
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:3b")

# ── Configure structured JSON logging ───────────────────────────────────────
logger = logging.getLogger("llm_observability")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))

file_handler = RotatingFileHandler(
    "logs/llm-calls.jsonl",
    maxBytes=10*1024*1024,
    backupCount=5
)

json_formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(llm_metrics)s",
    json_ensure_ascii=False,
    timestamp=True
)
file_handler.setFormatter(json_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def log_llm_call(prompt: str, model: str = None):
    """
    Now calls real Ollama model instead of mock.
    """
    model = model or DEFAULT_MODEL

    start_time = time.perf_counter()

    try:
        # ── Real LLM call via Ollama Python client ──────────────────────────
        response_obj = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.7,
                "num_predict": 512,           # max tokens to generate
            }
        )

        response_text = response_obj["message"]["content"]
        # Ollama returns token counts directly (very accurate)
        prompt_tokens = response_obj.get("prompt_eval_count", 0)
        completion_tokens = response_obj.get("eval_count", 0)

    except Exception as e:
        logger.error(f"Ollama call failed: {str(e)}")
        response_text = f"ERROR: Could not reach Ollama - {str(e)}"
        prompt_tokens = completion_tokens = 0

    latency_sec = time.perf_counter() - start_time

    metrics = {
        "model": model,
        "prompt": prompt[:300],                    # still truncate long ones
        "response": response_text[:500],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_seconds": round(latency_sec, 4),
        "hallucination_score": 0.00,               # placeholder
        "cost_usd": 0.000,                         # local = free
        "quality_score": 0.00                      # placeholder
    }

    # ── Log structured event ────────────────────────────────────────────────
    extra = {"llm_metrics": json.dumps(metrics)}
    logger.info(f"LLM call completed - model={model}", extra=extra)

    return response_text, metrics


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)

    prompts = [
        "What is the capital of France?",
        "Explain what is MLOps in one sentence like I'm a QA engineer with 2 years experience",
        "Write a funny one-liner joke about working from home in Bengaluru during monsoon"
    ]

    for p in prompts:
        print(f"\nPrompt: {p}")
        try:
            resp, met = log_llm_call(p)
            print(f"→ Model: {met['model']}")
            print(f"→ Response (first 120 chars): {resp[:120]}...")
            print(f"→ Latency: {met['latency_seconds']}s | Tokens: {met['total_tokens']}")
        except Exception as e:
            print(f"Error during call: {e}")