from __future__ import annotations
import subprocess
import sys
from typing import List
from rich.console import Console
from core.config import RecipeConfig

console = Console()


def build_fuse_command(cfg: RecipeConfig) -> List[str]:
    """Build the mlx_lm.fuse subprocess command. Verify flags with `mlx_lm.fuse --help`."""
    return [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", cfg.base_model,
        "--adapter-path", cfg.adapter_path,
        "--save-path", cfg.fused_path,
    ]


def fuse_adapter(cfg: RecipeConfig) -> None:
    """Merge LoRA adapter into base model weights. Raises RuntimeError on failure."""
    cmd = build_fuse_command(cfg)
    console.print(f"[bold blue]Fusing adapter:[/] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Fusion failed with exit code {result.returncode}")
    console.print(f"[bold blue]Fused model saved to:[/] {cfg.fused_path}")


def main() -> None:
    import argparse
    from core.config import load_recipe
    parser = argparse.ArgumentParser(description="Fuse LoRA adapter into base weights")
    parser.add_argument("--recipe", required=True, help="Path to recipe.yaml")
    args = parser.parse_args()
    cfg = load_recipe(args.recipe)
    fuse_adapter(cfg)


if __name__ == "__main__":
    main()
