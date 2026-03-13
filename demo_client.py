from rich.console import Console
from src.llm_client import LLMClient   # ← clean import thanks to __init__.py
from src.config import settings     # ← clean config import

console = Console()

if __name__ == "__main__":
    console.rule("[bold green]Session 5 Demo – LLMClient in Action[/bold green]")

    client =LLMClient(temperature=0.5)  # default provider/model from config
    
    client_ollama = LLMClient(provider="ollama", temperature=0.65)
    resp, met = client_ollama.explain_as_qa_engineer("pytest fixture")

    client_mock = LLMClient(provider="mock")
    resp_mock, met_mock = client_mock.generate_test_questions("date parsing")

    
    # Demo 1: QA Engineer personality
    console.print("\n[bold cyan]1. Explain 'pytest fixture' like a senior QA:[/bold cyan]")
    resp, met = client.explain_as_qa_engineer("pytest fixture")
    console.print(resp)
    console.print(f"[dim]Tokens: {met['total_tokens']} | Latency: {met['latency_seconds']:.3f}s | Session: {met['session_id']}[/dim]")
    
    # Demo 2: Dynamic test question generator
    console.print("\n[bold cyan]2. Generate test questions for date/time parsing:[/bold cyan]")
    resp, met = client.generate_test_questions("date and time parsing in Python")
    console.print(resp)
    console.print(f"[dim]Tokens: {met['total_tokens']} | Latency: {met['latency_seconds']:.3f}s[/dim]")
    
    console.rule("[bold green]✅ Session 5 Complete! Open logs/llm-calls.jsonl to see the new session_id field[/bold green]")