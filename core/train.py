from __future__ import annotations
import subprocess
import sys
from typing import List
from rich.console import Console
from core.config import RecipeConfig

console = Console()


def build_train_command(cfg: RecipeConfig) -> List[str]:
    """Build the mlx_lm.lora subprocess command from a RecipeConfig.

    Verify flag names against `mlx_lm.lora --help` in your installed version.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", cfg.base_model,
        "--train",
        "--data", cfg.data_dir,
        "--iters", str(cfg.iters),
        "--learning-rate", str(cfg.learning_rate),
        "--rank", str(cfg.lora_rank),
        "--lora-layers", str(cfg.lora_layers),
        "--batch-size", str(cfg.batch_size),
        "--adapter-path", cfg.adapter_path,
    ]
    if cfg.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    return cmd


def run_training(cfg: RecipeConfig) -> None:
    """Run MLX LoRA training. Raises RuntimeError if the subprocess fails."""
    cmd = build_train_command(cfg)
    console.print(f"[bold green]Starting training:[/] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    console.print("[bold green]Training complete.[/]")


def main() -> None:
    import argparse
    from core.config import load_recipe
    parser = argparse.ArgumentParser(description="Run MLX LoRA fine-tuning")
    parser.add_argument("--recipe", required=True, help="Path to recipe.yaml")
    args = parser.parse_args()
    cfg = load_recipe(args.recipe)
    run_training(cfg)


if __name__ == "__main__":
    main()
