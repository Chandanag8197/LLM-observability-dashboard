import pytest
from src.llm_client import LLMClient
from src.prompts import SYSTEM_QA_ENGINEER, SYSTEM_TEST_QUESTION_GENERATOR
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def mock_client():
    """Creates a fast mock-based client for every test"""
    return LLMClient(provider="mock", temperature=0.5)


def test_explain_as_qa_engineer_returns_string_and_metrics(mock_client):
    response, metrics = mock_client.explain_as_qa_engineer("pytest fixture")

    assert isinstance(response, str)
    assert "Mock explanation" in response   # mock backend behavior
    assert isinstance(metrics, dict)
    assert "provider" in metrics
    assert metrics["provider"] == "mock"
    assert "session_id" in metrics
    assert metrics["success"] is True
    assert "latency_seconds" in metrics
    assert metrics["latency_seconds"] > 0


def test_generate_test_questions_includes_cot_and_correct_system_prompt(mock_client):
    response, metrics = mock_client.generate_test_questions("datetime parsing", num=3)

    assert "Step 1:" in response or "FINAL ANSWER:" in response   # CoT was added
    assert "Mock test questions" in response
    assert metrics["chain_of_thought"] is True
    assert metrics["system_prompt_used"] is True
    assert metrics["prompt"].startswith("Create 3 high-quality")


def test_temperature_is_respected_in_metrics(mock_client):
    # Create another client with different temperature to compare
    hot_client = LLMClient(provider="mock", temperature=1.2)
    _, metrics_hot = hot_client.generate_test_questions("anything")

    assert metrics_hot["temperature"] == 1.2
    assert mock_client.temperature == 0.5   # fixture client


def test_metrics_always_have_required_keys(mock_client):
    _, metrics = mock_client.explain_as_qa_engineer("short concept")

    required_keys = {
        "provider", "model", "temperature", "prompt", "response",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "latency_seconds", "success", "session_id"
    }

    assert required_keys.issubset(metrics.keys())


@pytest.mark.parametrize("concept", ["fixture", "mock", "assertion"])
def test_explain_method_works_with_different_inputs(mock_client, concept):
    response, _ = mock_client.explain_as_qa_engineer(concept)
    assert len(response) > 10   # very basic check that we got something back