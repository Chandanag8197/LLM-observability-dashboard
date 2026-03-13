from typing import Tuple, Dict, Optional, Any
import time
import ollama
from .base import LLMBackend
from src.logger import logger


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_chain_of_thought: bool = False,
        **kwargs: Any
    ) -> Tuple[str, Dict]:
        start_time = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        final_prompt = prompt
        if use_chain_of_thought:
            cot = (
                "\n\nThink step by step before giving your final answer. "
                "Show your reasoning clearly, then put your final conclusion in this format:\n"
                "FINAL ANSWER: [your short answer here]"
            )
            final_prompt += cot

        messages.append({"role": "user", "content": final_prompt})

        try:
            response_obj = ollama.chat(
                model=self.model,
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
            "provider": self.provider_name,
            "model": self.model,
            "system_prompt_used": bool(system_prompt),
            "chain_of_thought": use_chain_of_thought,
            "temperature": temperature,
            "prompt": final_prompt[:400],
            "response": response_text[:800],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": round(latency_sec, 4),
            "success": prompt_tokens > 0 or "ERROR" not in response_text,
            # placeholders
            "hallucination_score": 0.0,
            "cost_usd": 0.0,
            "quality_score": 0.0,
        }

        return response_text, metrics