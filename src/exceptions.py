class LLMObservabilityError(Exception):
    """Base exception for our observability project."""
    pass


class LLMCallFailedError(LLMObservabilityError):
    """Raised when the underlying LLM provider (Ollama, Bedrock, etc.) fails."""
    def __init__(self, message, original_exception=None):
        self.original_exception = original_exception
        super().__init__(message)


class InvalidModelError(LLMObservabilityError):
    """Raised when an unsupported or invalid model name is provided."""
    pass


class MetricsValidationError(LLMObservabilityError):
    """Raised when collected metrics are incomplete or invalid."""
    pass