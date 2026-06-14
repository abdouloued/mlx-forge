"""mlx-forge — unified CLI entry point."""
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

COMMANDS = {
    "download": "Download a model from Hugging Face",
    "train":    "Fine-tune a model with LoRA",
    "eval":     "Score the trained adapter on validation data",
    "fuse":     "Merge adapter into full model weights",
    "loop":     "Run the auto-search ratchet loop overnight",
    "export":   "Convert fused model to GGUF for Ollama",
    "push":     "Upload fused model to Hugging Face Hub",
    "data":     "Validate or convert training data",
}


def show_help() -> None:
    console.print()
    console.print(Panel(
        "[bold white]Mac-native fine-tuning factory for open-weight models[/]\n"
        "[dim]Apple Silicon · MLX · LoRA · Evals-first[/]",
        title="[bold cyan]mlx-forge[/]",
        border_style="cyan",
        padding=(0, 4),
    ))
    console.print()

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="bold cyan", width=10)
    t.add_column(style="white")
    t.add_column(style="dim")

    t.add_row("download", "<model-id>",           COMMANDS["download"])
    t.add_row("train",    "--recipe <path>",       COMMANDS["train"])
    t.add_row("eval",     "--recipe <path>",       COMMANDS["eval"])
    t.add_row("fuse",     "--recipe <path>",       COMMANDS["fuse"])
    t.add_row("loop",     "--recipe <path>",       COMMANDS["loop"])
    t.add_row("export",   "--recipe <path> ...",   COMMANDS["export"])
    t.add_row("push",     "--recipe <path> --repo <id>", COMMANDS["push"])
    t.add_row("data",     "validate|convert ...",  COMMANDS["data"])

    console.print(t)

    console.print(
        "  [dim]Run [/][white]mlx-forge <command> --help[/][dim] for command-specific options.[/]\n"
        "  [dim]Docs: [/][white]github.com/abdouloued/mlx-forge[/]\n"
    )


def cmd_download(args: list[str]) -> None:
    from core.download import main as _main
    sys.argv = ["mlx-download"] + args
    _main()


def cmd_train(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="mlx-forge train")
    p.add_argument("--recipe", required=True, help="path to recipe.yaml")
    p.add_argument("--iters", type=int, help="override iters from recipe")
    p.add_argument("--rank",  type=int, help="override lora_rank from recipe")
    parsed = p.parse_args(args)

    from core.config import load_recipe
    import dataclasses
    cfg = load_recipe(parsed.recipe)
    overrides = {}
    if parsed.iters: overrides["iters"] = parsed.iters
    if parsed.rank:  overrides["lora_rank"] = parsed.rank
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    from core.train import run_training
    console.print(f"  [dim]Model:[/]   {cfg.base_model}")
    console.print(f"  [dim]Data:[/]    {cfg.data_dir}")
    console.print(f"  [dim]Iters:[/]   {cfg.iters}   rank={cfg.lora_rank}   layers={cfg.lora_layers}")
    console.print()
    run_training(cfg)
    console.print(f"\n  [green]✓ adapter saved to {cfg.adapter_path}[/]")


def cmd_eval(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="mlx-forge eval")
    p.add_argument("--recipe", required=True)
    parsed = p.parse_args(args)

    from core.config import load_recipe
    cfg = load_recipe(parsed.recipe)

    # defer to the recipe's own eval module
    import importlib, pathlib
    recipe_name = pathlib.Path(parsed.recipe).parent.name
    mod = importlib.import_module(f"recipes.{recipe_name}.eval")
    score = mod.evaluate(cfg.base_model, cfg.data_dir, adapter_path=cfg.adapter_path)
    console.print(f"\n  score = [bold green]{score:.4f}[/]")


def cmd_fuse(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="mlx-forge fuse")
    p.add_argument("--recipe", required=True)
    parsed = p.parse_args(args)

    from core.config import load_recipe
    from core.fuse import fuse_adapter
    cfg = load_recipe(parsed.recipe)
    fuse_adapter(cfg)
    console.print(f"\n  [green]✓ fused weights saved to {cfg.fused_path}[/]")


def cmd_loop(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="mlx-forge loop")
    p.add_argument("--recipe", required=True)
    p.add_argument("--n-experiments", type=int, default=10)
    p.add_argument("--target-score", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    parsed = p.parse_args(args)

    from core.loop import ratchet_loop
    ratchet_loop(
        recipe_path=parsed.recipe,
        n_experiments=parsed.n_experiments,
        target_score=parsed.target_score,
        seed=parsed.seed,
    )


def cmd_export(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="mlx-forge export")
    p.add_argument("--recipe",       required=True)
    p.add_argument("--llama-cpp",    required=True, dest="llama_cpp_dir")
    p.add_argument("--output",       required=True, dest="output_gguf")
    p.add_argument("--quantization", default="Q4_K_M")
    p.add_argument("--system",       default=None, dest="system_prompt")
    parsed = p.parse_args(args)

    from core.config import load_recipe
    from core.export_gguf import export_gguf
    cfg = load_recipe(parsed.recipe)
    export_gguf(
        fused_path=cfg.fused_path,
        output_gguf=parsed.output_gguf,
        llama_cpp_dir=parsed.llama_cpp_dir,
        quantization=parsed.quantization,
        system_prompt=parsed.system_prompt,
    )
    console.print(f"\n  [green]✓ {parsed.output_gguf}[/]")


def cmd_push(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="mlx-forge push")
    p.add_argument("--recipe",  required=True)
    p.add_argument("--repo",    required=True, dest="repo_id")
    p.add_argument("--public",  action="store_true")
    parsed = p.parse_args(args)

    from core.config import load_recipe
    from core.push_hf import push_to_hf
    cfg = load_recipe(parsed.recipe)
    push_to_hf(
        fused_path=cfg.fused_path,
        repo_id=parsed.repo_id,
        base_model=cfg.base_model,
        recipe=parsed.recipe,
        private=not parsed.public,
    )
    console.print(f"\n  [green]✓ pushed to huggingface.co/{parsed.repo_id}[/]")


def cmd_data(args: list[str]) -> None:
    from core.datakit import main as _main
    sys.argv = ["mlx-data"] + args
    _main()


DISPATCH = {
    "download": cmd_download,
    "train":    cmd_train,
    "eval":     cmd_eval,
    "fuse":     cmd_fuse,
    "loop":     cmd_loop,
    "export":   cmd_export,
    "push":     cmd_push,
    "data":     cmd_data,
}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        show_help()
        return

    cmd, *rest = args

    if cmd not in DISPATCH:
        console.print(f"\n  [red]Unknown command:[/] {cmd!r}\n")
        console.print(f"  Run [white]mlx-forge --help[/] to see available commands.\n")
        sys.exit(1)

    try:
        DISPATCH[cmd](rest)
    except KeyboardInterrupt:
        console.print("\n  [yellow]Interrupted.[/]")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"\n  [red]Error:[/] {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
