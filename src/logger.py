import time
import json
from datetime import datetime

def log_llm_call(prompt: str, model: str = "mock-llama3.2"):
    """First version of our metric logger"""
    start_time = time.time()
    
    # Mock LLM call (we replace this with real calls later)
    time.sleep(1.5)  # simulate network + inference latency
    response = f"Mock answer to: {prompt[:50]}..."
    
    latency = time.time() - start_time
    
    # Basic metrics we will expand every module
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "prompt_tokens": len(prompt.split()),      # rough estimate
        "completion_tokens": len(response.split()),
        "latency_seconds": round(latency, 3),
        "hallucination_score": 0.0,                # placeholder
        "cost_usd": 0.0,                           # placeholder
        "quality_score": 0.85                      # placeholder
    }
    
    print(json.dumps(metrics, indent=2))
    return response, metrics

# Test it
if __name__ == "__main__":
    prompt = "Explain MLOps in one sentence like I'm a QA engineer"
    response, metrics = log_llm_call(prompt)
    print(f"\nResponse: {response}")