"""
Survey Agent CLI Demo
─────────────────────
Interactive test harness that demonstrates all three agents end-to-end
without needing a running server. Uses the agents directly (bypasses HTTP).

Usage:
    python -m cli.demo                      # Full interactive demo
    python -m cli.demo design               # Design Agent demo only
    python -m cli.demo attempt              # Attempt Agent demo only
    python -m cli.demo insights             # Insight Agent demo only
    python -m cli.demo quality --file q.json  # Check questions from file
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

load_dotenv()

console = Console()
app = typer.Typer(help="Survey Agent CLI Demo")

# ─── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_SURVEY = {
    "title": "Telemedicine Platform Experience Survey",
    "description": "Gather feedback on doctors' experience with our telehealth tools",
    "questions": [
        {
            "id": "q1",
            "text": "How much do you love our new telemedicine platform?",
            "type": "likert",
            "options": ["1", "2", "3", "4", "5"],
            "required": True,
        },
        {
            "id": "q2",
            "text": "Don't you think video call quality has improved a lot?",
            "type": "boolean",
            "required": True,
        },
        {
            "id": "q3",
            "text": "What is your biggest pain point with the current telehealth workflow?",
            "type": "text",
            "required": False,
        },
        {
            "id": "q4",
            "text": "Do you prefer video or phone consultations and why?",
            "type": "mcq",
            "options": ["Video", "Phone"],
            "required": True,
        },
        {
            "id": "q5",
            "text": "What is your NPS for the telemedicine platform?",
            "type": "likert",
            "options": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "required": True,
        },
    ],
}

SAMPLE_RESPONSES = [
    {
        "answers": {
            "q1": 3,
            "q2": True,
            "q3": "Patients often can't figure out how to join the video call. We need a simpler join link.",
            "q4": "Video",
            "q5": 6,
        },
        "doctor_specialty": "Cardiology",
        "time_spent_seconds": 145,
    },
    {
        "answers": {
            "q1": 4,
            "q2": False,
            "q3": "Documentation is still a nightmare after telemedicine visits. The EHR doesn't auto-populate.",
            "q4": "Phone",
            "q5": 5,
        },
        "doctor_specialty": "Primary Care",
        "time_spent_seconds": 198,
    },
    {
        "answers": {
            "q1": 2,
            "q2": False,
            "q3": "Technical issues during calls. Patients drop frequently. Need better mobile support.",
            "q4": "Phone",
            "q5": 3,
        },
        "doctor_specialty": "Neurology",
        "time_spent_seconds": 210,
    },
    {
        "answers": {
            "q1": 5,
            "q2": True,
            "q3": "Overall good experience. Would love better scheduling integration with calendar.",
            "q4": "Video",
            "q5": 8,
        },
        "doctor_specialty": "Psychiatry",
        "time_spent_seconds": 120,
    },
    {
        "answers": {
            "q1": 3,
            "q2": True,
            "q3": "Patient consent forms need to be digital and integrated. Too much paperwork still.",
            "q4": "Video",
            "q5": 7,
        },
        "doctor_specialty": "Primary Care",
        "time_spent_seconds": 167,
    },
]

DOCTOR_CONTEXT = {
    "specialty": "Cardiology",
    "years_experience": 8,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_api_key() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            Panel(
                "[red]OPENAI_API_KEY not set.[/red]\n"
                "Copy [bold].env.example[/bold] → [bold].env[/bold] and add your key.",
                title="⚠️  Configuration Required",
                border_style="red",
            )
        )
        return False
    return True


def _spinner(label: str):
    return Progress(SpinnerColumn(), TextColumn(label), transient=True)


def _print_section(title: str, content: str, style: str = "cyan") -> None:
    console.print(Panel(content, title=f"[bold]{title}[/bold]", border_style=style))


# ─── Design Agent Demo ────────────────────────────────────────────────────────

async def _demo_design_agent() -> None:
    from app.agents.design_agent import design_agent

    console.print("\n[bold cyan]═══ DESIGN AGENT DEMO ═══[/bold cyan]")
    console.print("Analyzing a survey with intentional bias issues...\n")

    # Show input
    console.print("[dim]Input Survey:[/dim]")
    for i, q in enumerate(SAMPLE_SURVEY["questions"], 1):
        console.print(f"  [dim]{i}.[/dim] {q['text']}")
    console.print()

    # Quality check
    with _spinner("[yellow]Running quality check...[/yellow]") as progress:
        progress.add_task("", total=None)
        result = await design_agent.quality_check(
            survey_title=SAMPLE_SURVEY["title"],
            questions=SAMPLE_SURVEY["questions"],
            specialty="Mixed specialties",
        )

    # Quality score
    score_color = "green" if result.overall_quality_score >= 7 else "yellow" if result.overall_quality_score >= 5 else "red"
    console.print(f"[bold]Quality Score:[/bold] [{score_color}]{result.overall_quality_score:.1f}/10[/{score_color}]")
    console.print(f"[bold]Predicted Completion Rate:[/bold] [yellow]{result.estimated_completion_rate:.0f}%[/yellow]")
    console.print(f"[bold]Estimated Time:[/bold] {result.estimated_time_seconds}s ({result.estimated_time_seconds // 60}m {result.estimated_time_seconds % 60}s)\n")

    # Bias flags table
    if result.bias_flags:
        table = Table(title="🚩 Bias Flags", box=box.ROUNDED, show_lines=True)
        table.add_column("Question ID", style="dim")
        table.add_column("Type", style="red")
        table.add_column("Severity", justify="center")
        table.add_column("Original")
        table.add_column("Suggested Fix", style="green")

        severity_colors = {"high": "red", "medium": "yellow", "low": "dim"}
        for flag in result.bias_flags:
            sev_color = severity_colors.get(flag.severity, "white")
            table.add_row(
                flag.question_id,
                flag.bias_type.replace("_", " "),
                f"[{sev_color}]{flag.severity.upper()}[/{sev_color}]",
                flag.original_text[:60] + ("..." if len(flag.original_text) > 60 else ""),
                flag.suggestion[:60] + ("..." if len(flag.suggestion) > 60 else ""),
            )
        console.print(table)

    # Length recommendation
    console.print(f"\n[bold]Length Recommendation:[/bold] {result.length_recommendation}")
    if result.audience_suggestion:
        console.print(f"[bold]Audience Suggestion:[/bold] {result.audience_suggestion}")

    # Ask to generate variants
    console.print()
    if Confirm.ask("[bold]Generate A/B test variants?[/bold]", default=True):
        with _spinner("[yellow]Generating variants...[/yellow]") as progress:
            progress.add_task("", total=None)
            variants_result = await design_agent.generate_variants(
                title=SAMPLE_SURVEY["title"],
                questions=SAMPLE_SURVEY["questions"],
                num_variants=2,
            )

        for v in variants_result.variants:
            _print_section(
                f"Variant {v.variant_label} — Predicted: {v.predicted_completion_rate:.0f}%",
                f"[bold]Hypothesis:[/bold] {v.hypothesis}\n\n"
                + "[bold]Key Differences:[/bold]\n"
                + "\n".join(f"  • {d}" for d in v.key_differences)
                + f"\n\n[bold]Questions ({len(v.questions)}):[/bold]\n"
                + "\n".join(f"  {i}. {q.text}" for i, q in enumerate(v.questions)),
                style="blue" if v.variant_label == "A" else "magenta",
            )


# ─── Attempt Agent Demo ───────────────────────────────────────────────────────

async def _demo_attempt_agent() -> None:
    from app.agents.attempt_agent import attempt_agent

    console.print("\n[bold green]═══ ATTEMPT AGENT DEMO ═══[/bold green]")
    console.print("Simulating a doctor taking the survey...\n")

    # Progress messages
    console.print("[bold]── Progress Tracking ──[/bold]")
    for answered in [0, 2, 4, 5]:
        progress = await attempt_agent.get_progress(
            session_id="demo-session-001",
            questions_total=5,
            questions_answered=answered,
        )
        bar_filled = int(progress.percent_complete / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        console.print(
            f"  [{bar}] {progress.percent_complete:.0f}% "
            f"| {progress.estimated_seconds_remaining}s left "
            f"| {progress.motivational_message}"
        )

    console.print()

    # Clarification
    console.print("[bold]── Question Clarification ──[/bold]")
    confusing_question = SAMPLE_SURVEY["questions"][4]  # NPS question
    console.print(f"\n[dim]Doctor sees:[/dim] [italic]\"{confusing_question['text']}\"[/italic]")
    console.print("[dim]Doctor clicks \"Need help?\"...[/dim]\n")

    with _spinner("[yellow]Fetching clarification...[/yellow]") as progress:
        progress.add_task("", total=None)
        clarification = await attempt_agent.clarify_question(
            session_id="demo-session-001",
            question=confusing_question,
            doctor_context=DOCTOR_CONTEXT,
        )

    _print_section(
        "💡 AI Clarification",
        clarification.clarification
        + (
            "\n\n[bold]Examples:[/bold]\n"
            + "\n".join(f"  • {e}" for e in (clarification.examples or []))
            if clarification.examples else ""
        ),
        style="green",
    )
    console.print(
        f"[dim]✓ Meaning preserved (did_change_meaning={clarification.did_change_meaning})[/dim]\n"
    )

    # Completion summary
    if Confirm.ask("[bold]Show completion summary?[/bold]", default=True):
        console.print("\n[dim]Doctor completes survey...[/dim]")

        with _spinner("[yellow]Generating completion summary...[/yellow]") as progress:
            progress.add_task("", total=None)
            summary = await attempt_agent.generate_completion_summary(
                responses=list(SAMPLE_RESPONSES[0]["answers"].items()),
                survey_title=SAMPLE_SURVEY["title"],
                total_responses=247,
            )

        _print_section(
            "🎉 Completion",
            f"{summary.thank_you_message}\n\n"
            f"[bold]Community Insight:[/bold] {summary.aggregate_insight}\n\n"
            f"[bold]What's Next:[/bold] {summary.next_steps}",
            style="green",
        )


# ─── Insight Agent Demo ───────────────────────────────────────────────────────

async def _demo_insight_agent() -> None:
    from app.agents.insight_agent import insight_agent

    console.print("\n[bold magenta]═══ INSIGHT AGENT DEMO ═══[/bold magenta]")
    console.print(f"Analyzing {len(SAMPLE_RESPONSES)} survey responses...\n")

    survey_meta = {
        "id": "demo-survey-001",
        "title": SAMPLE_SURVEY["title"],
        "description": SAMPLE_SURVEY["description"],
        "questions": SAMPLE_SURVEY["questions"],
    }

    with _spinner("[yellow]Running insight analysis...[/yellow]") as progress:
        progress.add_task("", total=None)
        result = await insight_agent.analyze(
            survey_metadata=survey_meta,
            responses=SAMPLE_RESPONSES,
            completion_rate=68.4,
        )

    # Executive Summary
    _print_section("📊 Executive Summary", result.executive_summary, style="magenta")

    # Metrics
    metrics_table = Table(box=box.SIMPLE)
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("Completion Rate", f"{result.completion_rate:.1f}%")
    metrics_table.add_row(
        "Sentiment",
        f"✅ {result.sentiment_breakdown.get('positive', 0)*100:.0f}% positive | "
        f"❌ {result.sentiment_breakdown.get('negative', 0)*100:.0f}% negative | "
        f"➖ {result.sentiment_breakdown.get('neutral', 0)*100:.0f}% neutral",
    )
    console.print(metrics_table)

    # Themes
    console.print("\n[bold]🔍 Themes Identified:[/bold]")
    for i, theme in enumerate(result.themes, 1):
        sentiment_icon = {"positive": "✅", "negative": "❌", "neutral": "➖", "mixed": "🔄"}.get(
            theme.sentiment, "❓"
        )
        _print_section(
            f"{i}. {theme.title} {sentiment_icon} ({theme.prevalence_pct:.0f}% of responses)",
            theme.description
            + (
                "\n\n[dim]Representative themes:[/dim]\n"
                + "\n".join(f'  "{q}"' for q in (theme.representative_quotes or []))
                if theme.representative_quotes else ""
            ),
            style="white",
        )

    # Action Items
    console.print("\n[bold]✅ Action Items:[/bold]")
    priority_colors = {"high": "red", "medium": "yellow", "low": "dim"}
    for item in result.action_items:
        color = priority_colors.get(item.priority, "white")
        console.print(
            f"  [{color}][{item.priority.upper()}][/{color}] {item.description} "
            f"[dim]→ {item.owner_suggestion}[/dim]"
        )


# ─── CLI Commands ─────────────────────────────────────────────────────────────

@app.command()
def design():
    """Demo: Design Agent (survey quality check + variants)."""
    if not _check_api_key():
        raise typer.Exit(1)
    asyncio.run(_demo_design_agent())


@app.command()
def attempt():
    """Demo: Attempt Agent (clarification + progress + completion)."""
    if not _check_api_key():
        raise typer.Exit(1)
    asyncio.run(_demo_attempt_agent())


@app.command()
def insights():
    """Demo: Insight Agent (theme extraction + recommendations)."""
    if not _check_api_key():
        raise typer.Exit(1)
    asyncio.run(_demo_insight_agent())


@app.command()
def full():
    """Run the full end-to-end demo of all three agents."""
    if not _check_api_key():
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold]Survey Agent — End-to-End Demo[/bold]\n\n"
            "This demo runs all three AI agents:\n"
            "  1. [cyan]Design Agent[/cyan]   — Admin survey quality check + A/B variants\n"
            "  2. [green]Attempt Agent[/green]  — Doctor clarification + progress tracking\n"
            "  3. [magenta]Insight Agent[/magenta] — Post-survey theme + recommendation analysis",
            title="🤖 Welcome",
            border_style="bold white",
        )
    )
    console.print()

    async def _run_all():
        try:
            await _demo_design_agent()
            console.print("\n" + "─" * 60 + "\n")
            await _demo_attempt_agent()
            console.print("\n" + "─" * 60 + "\n")
            await _demo_insight_agent()
        finally:
            # Cleanly close Redis before event loop shuts down
            from app.redis_client import close_redis
            await close_redis()


    asyncio.run(_run_all())
    console.print(
        Panel(
            "[bold green]✅ Full demo complete![/bold green]\n"
            "Start the API server: [bold]uvicorn app.main:app --reload[/bold]\n"
            "API docs: [bold]http://localhost:8000/docs[/bold]",
            border_style="green",
        )
    )


@app.command()
def quality(
    file: Path = typer.Option(None, "--file", "-f", help="JSON file with questions array"),
):
    """Check questions from a JSON file for bias and quality."""
    if not _check_api_key():
        raise typer.Exit(1)

    if file and file.exists():
        data = json.loads(file.read_text())
        questions = data if isinstance(data, list) else data.get("questions", [])
        title = data.get("title", "Uploaded Survey") if isinstance(data, dict) else "Uploaded Survey"
    else:
        console.print("[dim]No file provided — using sample survey[/dim]")
        questions = SAMPLE_SURVEY["questions"]
        title = SAMPLE_SURVEY["title"]

    async def _run():
        from app.agents.design_agent import design_agent
        result = await design_agent.quality_check(title, questions)

        console.print(f"\n[bold]Quality Score:[/bold] {result.overall_quality_score:.1f}/10")
        console.print(f"[bold]Estimated Completion Rate:[/bold] {result.estimated_completion_rate:.0f}%")
        console.print(f"[bold]Estimated Time:[/bold] {result.estimated_time_seconds}s")
        console.print(f"[bold]Bias Flags:[/bold] {len(result.bias_flags)}")

        for flag in result.bias_flags:
            console.print(
                f"\n  [red]⚠[/red]  [{flag.severity.upper()}] {flag.bias_type}\n"
                f"     Original: {flag.original_text}\n"
                f"     Fix: [green]{flag.suggestion}[/green]"
            )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
