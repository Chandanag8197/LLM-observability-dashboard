import time
import json
import random
from datetime import datetime

def log_llm_call(prompt: str, model: str = "mock-llama3.2"):
    """First version of our metric logger"""
    start_time = time.time()
   
    # Simulate variable LLM latency
    latency_simulation = random.uniform(0.8, 3.0)
    time.sleep(latency_simulation)
    
    # Mock LLM response
    response = f"Mock answer to: {prompt[:50]}..."
   
    real_latency = time.time() - start_time
   
    # Basic metrics we will expand every module
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "prompt_tokens": len(prompt.split()),           # rough estimate
        "completion_tokens": len(response.split()),
        "latency_seconds": round(real_latency, 3),
        "simulated_latency_seconds": round(latency_simulation, 3),  # added for visibility
        "hallucination_score": 0.0,   # placeholder
        "cost_usd": 0.0,              # placeholder
        "quality_score": 0.85         # placeholder
    }
   
    print(json.dumps(metrics, indent=2))
    print("-" * 50)
    return response, metrics


if __name__ == "__main__":
    prompts = [
        "Explain MLOps in one sentence like I'm a QA engineer",
        "What is the difference between precision and recall? Explain like I'm 12 years old.",
        "Give me a funny one-liner about data scientists and coffee",
    ]

    print("Running 3 example LLM calls...\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nCall #{i}")
        print(f"Prompt: {prompt}")
        response, metrics = log_llm_call(prompt)
        print(f"Response: {response}\n")