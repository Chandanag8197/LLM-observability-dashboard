from rich.console import Console
from src.llm_client import LLMClient   # ← clean import thanks to __init__.py

console = Console()

if __name__ == "__main__":
    console.rule("[bold green]Session 5 Demo – LLMClient in Action[/bold green]")
    
    client = LLMClient(temperature=0.65)
    
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