import time
from typing import Tuple, Dict, Optional, Any
from .base import LLMBackend


class MockBackend(LLMBackend):
    """Fake LLM that returns instant deterministic or random-like answers – great for tests & CI"""

    def __init__(self, model: str = "mock-model"):
        self.model = model

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_chain_of_thought: bool = False,
        **kwargs: Any
    ) -> Tuple[str, Dict]:
        start = time.perf_counter()

        # Very simple deterministic response + some fake variation
        if "explain" in prompt.lower() or "what is" in prompt.lower():
            answer = f"Mock explanation of '{prompt}' in QA engineer style."
        elif "test question" in prompt.lower() or "create" in prompt.lower():
            answer = "Mock test questions:\n1. Edge case question?\n2. Tricky question?"
        else:
            answer = f"Mock response to: {prompt[:60]}..."

        if use_chain_of_thought:
            answer = f"Step 1: Think...\nStep 2: Reason...\nFINAL ANSWER: {answer}"

        latency = 0.08 + (len(prompt) / 2000)   # fake ~80–300 ms

        metrics = {
            "provider": "mock",
            "model": self.model,
            "system_prompt_used": bool(system_prompt),
            "chain_of_thought": use_chain_of_thought,
            "temperature": temperature,
            "prompt": prompt[:400],
            "response": answer[:800],
            "prompt_tokens": len(prompt) // 4 + 10,
            "completion_tokens": len(answer) // 4 + 5,
            "total_tokens": 0,  # filled below
            "latency_seconds": round(time.perf_counter() - start + latency, 4),
            "success": True,
            "hallucination_score": 0.0,
            "cost_usd": 0.0,
            "quality_score": 0.95,
        }
        metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

        time.sleep(latency)  # simulate network / inference time

        return answer, metrics