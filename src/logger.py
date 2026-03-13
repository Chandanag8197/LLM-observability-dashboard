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
import sys
from pathlib import Path
from src.config import settings  # clean config import

# Force-add project root to Python's search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()                       # reads .env file
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:3b")

# ── Configure structured JSON logging ───────────────────────────────────────
logger = logging.getLogger("llm_observability")
logger.setLevel(settings.log_level_int)  # from config.py, supports env var override

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))

file_handler = RotatingFileHandler(
    settings.metrics_file,
    maxBytes=settings.metrics_max_bytes,
    backupCount=settings.metrics_backup_count
)

json_formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(llm_metrics)s",
    json_ensure_ascii=False,
    timestamp=True
)
file_handler.setFormatter(json_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def log_llm_call(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    use_chain_of_thought: bool = False
):
    model = model or settings.default_model

    start_time = time.perf_counter()

    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Optional: add CoT instruction
    if use_chain_of_thought:
        cot_instruction = (
            "\n\nThink step by step before giving your final answer. "
            "Show your reasoning clearly, then put your final conclusion in this format:\n"
            "FINAL ANSWER: [your short answer here]"
        )
        prompt += cot_instruction

    messages.append({"role": "user", "content": prompt})

    try:
        response_obj = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        )

        response_text = response_obj["message"]["content"]
        prompt_tokens = response_obj.get("prompt_eval_count", 0)
        completion_tokens = response_obj.get("eval_count", 0)

    except Exception as e:
        logger.error(f"Ollama call failed: {str(e)}")
        response_text = f"ERROR: {str(e)}"
        prompt_tokens = completion_tokens = 0

    latency_sec = time.perf_counter() - start_time

    metrics = {
        "model": model,
        "system_prompt_used": bool(system_prompt),
        "chain_of_thought": use_chain_of_thought,
        "temperature": temperature,
        "prompt": prompt[:400],
        "response": response_text[:800],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_seconds": round(latency_sec, 4),
        "hallucination_score": 0.00,  # placeholder
        "cost_usd": 0.000,
        "quality_score": 0.00
    }

    extra = {"llm_metrics": json.dumps(metrics)}
    logger.info(f"LLM call - model={model} | CoT={use_chain_of_thought}", extra=extra)

    return response_text, metrics


# ── Improved test block ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console
    from src.prompts import SYSTEM_QA_ENGINEER, SYSTEM_TEST_QUESTION_GENERATOR

    console = Console()
    os.makedirs("logs", exist_ok=True)

    examples = [
        {
            "name": "Simple QA (no system)",
            "prompt": "What is pytest fixture?",
            "system": None,
            "cot": False
        },
        {
            "name": "With QA Engineer personality",
            "prompt": "What is pytest fixture?",
            "system": SYSTEM_QA_ENGINEER,
            "cot": False
        },
        {
            "name": "Generate test questions + CoT",
            "prompt": "Create test questions to check if an LLM understands date and time parsing correctly.",
            "system": SYSTEM_TEST_QUESTION_GENERATOR,
            "cot": True
        }
    ]

    for ex in examples:
        console.rule(f"[bold green]{ex['name']}[/bold green]")
        console.print(f"[bold]Prompt:[/bold] {ex['prompt']}\n")

        resp, met = log_llm_call(
            prompt=ex["prompt"],
            system_prompt=ex["system"],
            use_chain_of_thought=ex["cot"],
            temperature=0.65
        )

        console.print("[bold cyan]Response:[/bold cyan]")
        console.print(resp)
        console.print(f"\n[dim]Latency: {met['latency_seconds']:.3f}s | Tokens: {met['total_tokens']}[/dim]\n")