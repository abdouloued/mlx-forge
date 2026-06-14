"""Download and cache a model from Hugging Face via mlx-lm."""
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def download_model(model_id: str) -> Path:
    from mlx_lm import load
    from huggingface_hub import snapshot_download
    import os

    console.print(f"\n  [dim]Model:[/] [white]{model_id}[/]")
    console.print(f"  [dim]Cache:[/] [white]{Path.home()}/.cache/huggingface/hub/[/]\n")

    start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Downloading model weights …", total=None)

        # snapshot_download pulls all shards and shows individual file progress
        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        progress.update(task, description="Loading model to verify …")

        # load() triggers MLX conversion/cache if needed
        load(model_id)
        progress.update(task, description="[green]Done.[/]", completed=True)

    elapsed = time.time() - start
    size_gb = sum(
        f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()
    ) / 1e9

    console.print(Panel(
        f"  [green]✓[/] [white]{model_id}[/]\n"
        f"  [dim]Path:[/]    {local_dir}\n"
        f"  [dim]Size:[/]    {size_gb:.1f} GB\n"
        f"  [dim]Time:[/]    {elapsed:.0f}s",
        title="[bold white]model downloaded[/]",
        border_style="green",
        padding=(0, 2),
    ))
    return Path(local_dir)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        console.print(
            "\n[bold white]Usage:[/]\n"
            "  uv run python -m core.download [bold cyan]<model-id>[/]\n\n"
            "[bold white]Examples:[/]\n"
            "  uv run python -m core.download mlx-community/Qwen2.5-7B-Instruct-4bit\n"
            "  uv run python -m core.download mlx-community/Phi-4-mini-instruct-4bit\n"
            "  uv run python -m core.download mlx-community/Llama-3.2-3B-Instruct-4bit\n\n"
            "[dim]Models are cached at ~/.cache/huggingface/hub/ and reused on future runs.[/]\n"
        )
        sys.exit(0)

    model_id = sys.argv[1]
    try:
        download_model(model_id)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[red]Error:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
