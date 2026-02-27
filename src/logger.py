import time
import json
import logging
import random
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pythonjsonlogger import jsonlogger


# ── Configure structured JSON logging ───────────────────────────────────────
logger = logging.getLogger("llm_observability")
logger.setLevel(logging.INFO)

# Console handler (human readable for local dev)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))

# File handler → JSON lines (perfect for Splunk ingestion later)
file_handler = RotatingFileHandler(
    "logs/llm-calls.jsonl",
    maxBytes=10*1024*1024,      # 10 MB per file
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


def log_llm_call(prompt: str, model: str = "mock-llama3.2"):
    """
    Production-style LLM call logger.
    Returns: (response_text, metrics_dict)
    """
    start_time = time.perf_counter()           # higher precision than time.time()

    # ── Mock LLM response (later → real API / local model) ──────────────────
    time.sleep(random.uniform(0.4, 3.1))    # fake 0.7–2.3s latency
    response = f"Mock intelligent reply to: {prompt[:60]}..."

    latency_sec = time.perf_counter() - start_time

    # Rough token estimation (good enough for v1)
    prompt_tokens = len(prompt.split()) + len(prompt) // 4
    completion_tokens = len(response.split()) + len(response) // 4

    metrics = {
        "model": model,
        "prompt": prompt[:200],                    # truncate long prompts
        "response": response,
        "prompt_tokens_estimated": prompt_tokens,
        "completion_tokens_estimated": completion_tokens,
        "total_tokens_estimated": prompt_tokens + completion_tokens,
        "latency_seconds": round(latency_sec, 4),
        "hallucination_score": 0.00,               # placeholder
        "cost_usd": 0.000,                         # placeholder
        "quality_score": 0.92                      # placeholder
    }

    # ── Log structured event ────────────────────────────────────────────────
    extra = {"llm_metrics": json.dumps(metrics)}   # trick to inject nested JSON
    logger.info("LLM call completed", extra=extra)

    return response, metrics


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nCheck logs/llm-calls.jsonl — we just created production-style structured logs!")
    import os
    os.makedirs("logs", exist_ok=True)          # create logs/ folder

    prompts = [
        "What is the capital of France?",
        "Explain CI/CD pipeline like I'm a 2-year experienced QA engineer",
        "Write a funny one-liner about Bengaluru traffic"
    ]

    for p in prompts:
        print(f"\nPrompt: {p}")
        resp, met = log_llm_call(p)
        print(f"→ Response: {resp[:80]}...")