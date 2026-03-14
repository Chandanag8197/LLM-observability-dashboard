from typing import Optional, Tuple, Dict
import uuid
from src.config import settings
from src.exceptions import InvalidModelError, LLMCallFailedError
from src.llm_backends.ollama_backend import OllamaBackend
from src.llm_backends.mock_backend import MockBackend
from src.prompts import SYSTEM_QA_ENGINEER, SYSTEM_TEST_QUESTION_GENERATOR
# from src.llm_backends.base import LLMBackend   # uncomment later when needed

class LLMClient:
    BACKEND_REGISTRY = {
        "ollama": OllamaBackend,
        "mock": MockBackend,
        # "openai": OpenAIBackend,     # future
        # "bedrock": BedrockBackend,
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        if provider not in self.BACKEND_REGISTRY:
            raise InvalidModelError(f"Unknown provider: {provider}")

        backend_class = self.BACKEND_REGISTRY[provider]
        self.backend = backend_class(model=model or settings.default_model)

        self.temperature = temperature if temperature is not None else settings.default_temperature
        self.session_id = str(uuid.uuid4())[:8]
        self.provider = provider

        print(f"LLMClient ready | Provider: {self.provider} | Model: {self.backend.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_chain_of_thought: bool = False,
        max_tokens: int = 512,
    ) -> Tuple[str, Dict]:
        try:
            response, metrics = self.backend.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=max_tokens,
                use_chain_of_thought=use_chain_of_thought,
            )
            metrics["session_id"] = self.session_id
            metrics["provider"] = self.provider

            # ── NEW: automatically log full metrics ────────────────────────
            from src.logger import log_llm_metrics          # ← add this import
            log_llm_metrics(metrics)                        # ← add this line

            return response, metrics

        except Exception as e:
            raise LLMCallFailedError(
                f"Backend {self.provider} failed", original_exception=e
            )

    def explain_as_qa_engineer(self, concept: str) -> Tuple[str, Dict]:
        """Uses your senior QA personality – perfect for documentation or training."""
        prompt = f"Explain the concept of '{concept}' clearly."
        return self.generate(prompt, system_prompt=SYSTEM_QA_ENGINEER)

    def generate_test_questions(self, topic: str, num: int = 4) -> Tuple[str, Dict]:
        """Uses your test-question generator + CoT – exactly like writing test cases!"""
        prompt = f"Create {num} high-quality, tricky test questions about: {topic}"
        return self.generate(prompt, system_prompt=SYSTEM_TEST_QUESTION_GENERATOR, use_chain_of_thought=True)