"""
mlx-forge demo — simulated pipeline for screen recording.

Run:  uv run python demo.py
Record with QuickTime (File → New Screen Recording) or:
  brew install asciinema && asciinema rec demo.cast
  asciinema play demo.cast
"""
import json
import subprocess
import time
import sys
import urllib.request
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.columns import Columns
from rich.padding import Padding

console = Console(width=88)

FAST = "--fast" in sys.argv   # uv run python demo.py --fast skips sleeps


def sleep(s: float) -> None:
    if not FAST:
        time.sleep(s)


def typewrite(text: str, delay: float = 0.035) -> None:
    for ch in text:
        console.print(ch, end="", highlight=False)
        if not FAST:
            time.sleep(delay)
    console.print()


def prompt(cmd: str) -> None:
    sleep(0.6)
    console.print(f"[bold green]❯[/] [bold white]{cmd}[/]")
    sleep(0.4)


# ── Banner ─────────────────────────────────────────────────────────────────────

def banner() -> None:
    console.clear()
    sleep(0.3)
    art = Text(justify="center")
    art.append("\n  mlx-forge\n", style="bold white")
    art.append("  Mac-native fine-tuning factory for open-weight models\n", style="dim white")
    art.append("  Apple Silicon · MLX · LoRA · Evals-first\n", style="dim cyan")
    console.print(Panel(art, border_style="bright_cyan", padding=(0, 4)))
    sleep(1.2)


# ── Step 1: validate data ──────────────────────────────────────────────────────

def demo_validate() -> None:
    console.print(Rule("[bold cyan]Step 1 — validate training data[/]", style="cyan"))
    sleep(0.5)
    prompt("uv run python -m core.datakit validate recipes/toolcalling/data/train.jsonl")
    sleep(0.5)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        t = p.add_task("Reading train.jsonl …", total=None)
        sleep(1.2)
        p.remove_task(t)

    console.print("  Total lines: [white]20[/]")
    console.print("    Valid:  [green]20[/]")
    console.print("    Errors: [green]0[/]")
    console.print()
    sleep(0.8)

    prompt("uv run python -m core.datakit validate recipes/toolcalling/data/valid.jsonl")
    sleep(0.4)
    console.print("  Total lines: [white]8[/]  │  Valid: [green]8[/]  │  Errors: [green]0[/]")
    sleep(1.0)
    console.print()


# ── Step 2: ratchet loop ───────────────────────────────────────────────────────

EXPERIMENTS = [
    # (lr, rank, layers, batch, iters, score, kept)
    (1e-4, 8,  16, 4, 500, 0.6875, True),
    (2e-4, 8,  16, 4, 500, 0.5625, False),
    (1e-4, 16, 16, 4, 500, 0.7500, True),
    (1e-4, 16, 24, 4, 500, 0.8125, True),
    (1e-4, 32, 24, 2, 500, 0.7812, False),
    (5e-5, 16, 24, 4, 500, 0.8750, True),
]


def score_color(s: float) -> str:
    if s >= 0.85:
        return "bold green"
    if s >= 0.70:
        return "yellow"
    return "red"


def demo_loop() -> None:
    console.print(Rule("[bold cyan]Step 2 — auto-search ratchet loop[/]", style="cyan"))
    sleep(0.5)
    prompt("uv run python -m core.loop --recipe recipes/toolcalling/recipe.yaml --n-experiments 6")
    sleep(0.8)
    console.print(f"  [dim]Search space: learning_rate · lora_rank · lora_layers · batch_size[/]")
    console.print(f"  [dim]Model: mlx-community/Qwen2.5-7B-Instruct-4bit[/]")
    console.print()
    sleep(0.8)

    best = 0.0

    results_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    results_table.add_column("Exp", style="dim", width=4)
    results_table.add_column("lr", width=7)
    results_table.add_column("rank", width=5)
    results_table.add_column("layers", width=6)
    results_table.add_column("batch", width=5)
    results_table.add_column("score", width=7)
    results_table.add_column("", width=14, no_wrap=True)

    for i, (lr, rank, layers, batch, iters, score, kept) in enumerate(EXPERIMENTS, 1):
        console.print(f"  [bold]Experiment {i}/6[/]  lr={lr:.0e}  rank={rank}  layers={layers}  batch={batch}")

        with Progress(
            SpinnerColumn(),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.description}[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as p:
            task = p.add_task(f"training {iters} iters …", total=iters)
            chunk = iters // 20
            for _ in range(20):
                sleep(0.09 if not FAST else 0.0)
                p.advance(task, chunk)
            sleep(0.1)

        sleep(0.2)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      console=console, transient=True) as p:
            t = p.add_task("scoring on valid.jsonl …", total=None)
            sleep(0.6 if not FAST else 0.0)
            p.remove_task(t)

        delta = score - best
        if kept:
            tag = f"[bold green]✓ +{delta:.4f}[/]"
            best = score
        else:
            tag = f"[dim red]✗ ({delta:+.4f})[/]"

        results_table.add_row(
            str(i),
            f"{lr:.0e}", str(rank), str(layers), str(batch),
            f"[{score_color(score)}]{score:.4f}[/]",
            tag,
        )

        console.print(f"  score=[{score_color(score)}]{score:.4f}[/]  {tag}")
        if kept:
            console.print(f"  [green]→ new best — git commit[/]")
        console.print()
        sleep(0.5 if not FAST else 0.0)

    console.print(results_table)
    sleep(0.6)
    console.print(f"\n  [bold green]Loop complete.[/]  Best score: [bold green]{best:.4f}[/]  "
                  f"(lr=5e-5  rank=16  layers=24  batch=4)")
    console.print()
    sleep(1.0)


# ── Step 3: fuse ───────────────────────────────────────────────────────────────

def demo_fuse() -> None:
    console.print(Rule("[bold cyan]Step 3 — fuse adapter into full weights[/]", style="cyan"))
    sleep(0.5)
    prompt("uv run python -m core.fuse --recipe recipes/toolcalling/recipe.yaml")
    sleep(0.4)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        t = p.add_task("merging LoRA adapter into Qwen2.5-7B …", total=None)
        sleep(2.0 if not FAST else 0.2)
        p.remove_task(t)

    console.print("  [green]✓ fused weights saved to fused/toolcalling/[/]")
    console.print()
    sleep(0.8)


# ── Step 4: export GGUF ────────────────────────────────────────────────────────

def demo_export() -> None:
    console.print(Rule("[bold cyan]Step 4 — export GGUF for Ollama[/]", style="cyan"))
    sleep(0.5)
    prompt("uv run python -m core.export_gguf \\\n"
           "  --fused-path fused/toolcalling \\\n"
           "  --output-gguf exports/toolcalling/model-q4_k_m.gguf \\\n"
           "  --llama-cpp-dir ~/llama.cpp --quantization Q4_K_M")
    sleep(0.4)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        t = p.add_task("convert_hf_to_gguf.py …", total=None)
        sleep(1.4 if not FAST else 0.1)
        p.update(t, description="llama-quantize Q4_K_M …")
        sleep(1.2 if not FAST else 0.1)
        p.remove_task(t)

    console.print("  [green]✓ exports/toolcalling/model-q4_k_m.gguf[/]")
    console.print("  [green]✓ exports/toolcalling/Modelfile[/]")
    console.print()
    sleep(0.8)

    prompt("cd exports/toolcalling && ollama create qwen-toolcalling -f Modelfile")
    sleep(0.5)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        t = p.add_task("registering model with Ollama …", total=None)
        sleep(1.0 if not FAST else 0.1)
        p.remove_task(t)
    console.print("  [green]✓ model 'qwen-toolcalling' ready[/]")
    console.print()
    sleep(0.8)


# ── Step 5: inference — real ollama call via REST API ─────────────────────────

_OLLAMA_MODEL = "qwen3.5:2b"
_OLLAMA_URL = "http://localhost:11434/api/chat"

_SYSTEM = (
    "You are a tool-calling assistant. Respond ONLY with a valid JSON object — "
    "no prose, no markdown fences, no explanation. Format:\n"
    '{"name": "<function_name>", "arguments": {"key": "value"}}'
)
_USER = "Book a flight from Paris to Tokyo on June 20."


def _call_ollama(model: str, system: str, user: str) -> str:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "/no_think " + user},
        ],
    }).encode()
    req = urllib.request.Request(
        _OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip()


def _pretty_json(text: str) -> str:
    """Return indented JSON if parseable, else strip markdown fences and return as-is."""
    import re
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.dumps(json.loads(text), indent=2)
    except json.JSONDecodeError:
        return text


def demo_inference() -> None:
    console.print(Rule("[bold cyan]Step 5 — run the model (live)[/]", style="cyan"))
    sleep(0.5)
    console.print(
        f"  [dim]Using [/][white]{_OLLAMA_MODEL}[/][dim] "
        "— stand-in for the fine-tuned adapter[/]"
    )
    console.print()
    sleep(0.4)
    prompt(f'ollama run {_OLLAMA_MODEL} "Book a flight from Paris to Tokyo on June 20"')
    sleep(0.4)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  console=console, transient=True) as p:
        t = p.add_task("running inference …", total=None)
        try:
            raw = _call_ollama(_OLLAMA_MODEL, _SYSTEM, _USER)
            output = _pretty_json(raw)
        except Exception as exc:
            output = f"(inference error: {exc})"
        p.remove_task(t)

    for line in output.splitlines():
        sleep(0.06 if not FAST else 0.0)
        console.print(f"  [bold cyan]{line}[/]")

    console.print()
    sleep(1.2)


# ── Summary ────────────────────────────────────────────────────────────────────

def demo_summary() -> None:
    console.print(Rule(style="bright_cyan"))
    sleep(0.4)

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim cyan", width=22)
    t.add_column(style="white")
    t.add_row("Model",          "Qwen2.5-7B-Instruct (4-bit)")
    t.add_row("Adapter",        "LoRA  rank=16  layers=24")
    t.add_row("Training",       "6 experiments × 500 iters each")
    t.add_row("Score (start)",  "[yellow]0.6875[/]")
    t.add_row("Score (final)",  "[bold green]0.8750[/]  (+27%)")
    t.add_row("Hardware",       "Apple Silicon · MLX · no CUDA")
    t.add_row("Publishing",     "manual — you decide when it ships")

    console.print(Panel(
        t,
        title="[bold white]mlx-forge — session summary[/]",
        border_style="bright_cyan",
        padding=(0, 2),
    ))
    sleep(0.6)
    console.print()
    console.print("  [bold white]github.com/abdouloued/mlx-forge[/]  ·  [dim]star ⭐ if this saves you time[/]")
    console.print()
    sleep(1.0)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    banner()
    demo_validate()
    demo_loop()
    demo_fuse()
    demo_export()
    demo_inference()
    demo_summary()
