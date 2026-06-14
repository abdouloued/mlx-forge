from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import yaml
from rich.console import Console

from core.config import RecipeConfig

console = Console()


def build_train_command(cfg: RecipeConfig) -> List[str]:
    """Build the mlx_lm lora subprocess command for mlx-lm 0.30+.

    In 0.31+, --rank and --lora-layers were removed from the CLI.
    LoRA rank/alpha go in a YAML config passed with -c; layers uses --num-layers.
    """
    # Write LoRA-specific params to a temp YAML config that mlx_lm reads via -c
    lora_cfg: dict = {
        "lora_parameters": {
            "rank": cfg.lora_rank,
            "alpha": cfg.lora_alpha,
            "dropout": 0.0,
            "scale": 10.0,
        }
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="mlxforge_lora_"
    )
    yaml.safe_dump(lora_cfg, tmp)
    tmp.flush()
    tmp.close()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",         cfg.base_model,
        "--train",
        "--data",          cfg.data_dir,
        "--iters",         str(cfg.iters),
        "--learning-rate", str(cfg.learning_rate),
        "--num-layers",    str(cfg.lora_layers),
        "--batch-size",    str(cfg.batch_size),
        "--adapter-path",  cfg.adapter_path,
        "-c",              tmp.name,
    ]
    if cfg.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    return cmd


def run_training(cfg: RecipeConfig) -> None:
    """Run MLX LoRA training. Raises RuntimeError if the subprocess fails."""
    cmd = build_train_command(cfg)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


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
