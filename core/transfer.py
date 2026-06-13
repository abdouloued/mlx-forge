"""
Transfer mode for mlx-forge.

The loop runs on a small sibling model (loop_model, 2-4B) to find the best
hyperparameters cheaply. This module then applies those hyperparams in a single
training run on the large ship model (base_model, 27B+).

Usage:
  uv run python -m core.transfer \\
    --recipe recipes/toolcalling/recipe.yaml \\
    --state-path loop_state.json

Workflow:
  1. Run core.loop on a recipe with mode=transfer (trains loop_model)
  2. Once the loop finds a good config, run core.transfer to train base_model once
"""
from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path

from rich.console import Console

from core.config import RecipeConfig, load_recipe
from core.loop import LoopState, load_state
from core.train import run_training

console = Console()

_TRANSFERABLE_PARAMS = ("learning_rate", "lora_rank", "lora_alpha", "lora_layers", "batch_size")


def build_ship_config(recipe_path: str, state_path: str) -> RecipeConfig:
    """
    Combine the ship model (base_model from recipe) with the best hyperparams
    found by the loop (stored in loop_state.json).
    """
    cfg = load_recipe(recipe_path)
    state: LoopState = load_state(state_path)

    overrides = {
        k: v
        for k, v in state.best_config.items()
        if k in _TRANSFERABLE_PARAMS
    }
    return dataclasses.replace(cfg, **overrides)


def apply_to_ship_model(recipe_path: str, state_path: str) -> RecipeConfig:
    """
    Apply the best loop config to the large ship model and run one training pass.

    Raises ValueError if the recipe is not in transfer mode — transfer is only
    meaningful when the loop ran on a different (smaller) model.
    """
    cfg = load_recipe(recipe_path)
    if cfg.mode != "transfer":
        raise ValueError(
            f"Recipe must use transfer mode; got mode={cfg.mode!r}. "
            "Set mode: transfer, base_model to your large ship model, "
            "and loop_model to the small sibling used for the search."
        )

    ship_cfg = build_ship_config(recipe_path, state_path)
    state = load_state(state_path)

    console.print(
        f"[bold cyan]Transfer mode:[/] applying loop best config "
        f"(score={state.best_score:.4f}, exp={state.experiment}) "
        f"to ship model [bold]{ship_cfg.base_model}[/]"
    )
    console.print(f"  lr={ship_cfg.learning_rate}  rank={ship_cfg.lora_rank}  "
                  f"layers={ship_cfg.lora_layers}  batch={ship_cfg.batch_size}")

    run_training(ship_cfg)
    console.print(f"[bold cyan]Ship model adapter saved to:[/] {ship_cfg.adapter_path}")
    return ship_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply best loop hyperparams to the large ship model (transfer mode)"
    )
    parser.add_argument("--recipe", required=True, help="Path to recipe.yaml (mode: transfer)")
    parser.add_argument("--state-path", default="loop_state.json",
                        help="Path to loop_state.json written by core.loop")
    args = parser.parse_args()
    apply_to_ship_model(args.recipe, args.state_path)


if __name__ == "__main__":
    main()
