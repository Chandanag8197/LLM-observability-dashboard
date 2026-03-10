from typing import Optional, Tuple, Dict
import uuid
from src.logger import log_llm_call          # ← uses your existing function!
from src.prompts import SYSTEM_QA_ENGINEER, SYSTEM_TEST_QUESTION_GENERATOR

class LLMClient:
    """Your reusable LLM 'test framework' class – clean, traceable, production-ready!"""
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7):
        self.model = model or "llama3.2:3b"
        self.temperature = temperature
        self.session_id = str(uuid.uuid4())[:8]   # unique ID for Splunk tracing later!
        
        print(f"✅ LLMClient ready! | Model: {self.model} | Session ID: {self.session_id}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 use_chain_of_thought: bool = False, max_tokens: int = 512) -> Tuple[str, Dict]:
        """Core method – calls your logger automatically and adds session tracking."""
        response, metrics = log_llm_call(
            prompt=prompt,
            model=self.model,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            use_chain_of_thought=use_chain_of_thought
        )
        metrics["session_id"] = self.session_id   # extra observability for future Splunk
        return response, metrics

    def explain_as_qa_engineer(self, concept: str) -> Tuple[str, Dict]:
        """Uses your senior QA personality – perfect for documentation or training."""
        prompt = f"Explain the concept of '{concept}' clearly."
        return self.generate(prompt, system_prompt=SYSTEM_QA_ENGINEER)

    def generate_test_questions(self, topic: str, num: int = 4) -> Tuple[str, Dict]:
        """Uses your test-question generator + CoT – exactly like writing test cases!"""
        prompt = f"Create {num} high-quality, tricky test questions about: {topic}"
        return self.generate(prompt, system_prompt=SYSTEM_TEST_QUESTION_GENERATOR, use_chain_of_thought=True)