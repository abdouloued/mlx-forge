"""
Auto-search ratchet loop for mlx-forge.

Pattern inspired by Karpathy's autoresearch: propose -> train -> score -> keep if better -> repeat.

For fine-tuning this is a modest hyperparameter sweep, not a 100-experiments research org.
The loop edits only the fine-tuning config -- never eval.py or core/.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from core.config import RecipeConfig, load_recipe
from core.train import run_training

console = Console()

SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
    "lora_rank": [4, 8, 16, 32],
    "lora_layers": [8, 16, 24, 32],
    "batch_size": [2, 4, 8],
}


@dataclass
class LoopState:
    best_score: float = 0.0
    experiment: int = 0
    best_config: dict = field(default_factory=dict)


def propose_config(cfg: RecipeConfig, search_space: dict, rng: random.Random) -> RecipeConfig:
    """Return a new RecipeConfig with exactly one hyperparameter varied."""
    param = rng.choice(list(search_space.keys()))
    value = rng.choice(search_space[param])
    return dataclasses.replace(cfg, **{param: value})


def save_state(state: LoopState, path: Path | str) -> None:
    Path(path).write_text(json.dumps(dataclasses.asdict(state), indent=2))


def load_state(path: Path | str) -> LoopState:
    p = Path(path)
    if not p.exists():
        return LoopState()
    data = json.loads(p.read_text())
    return LoopState(**data)


def update_program_md(path: Path | str, score: float, experiment: int) -> None:
    """Update the 'Current best score' block in program.md."""
    p = Path(path)
    content = p.read_text()
    content = re.sub(r"score: [\d.]+", f"score: {score:.4f}", content)
    content = re.sub(r"experiment: \S+", f"experiment: {experiment}", content)
    p.write_text(content)


def _git_commit(files: list[str], message: str) -> None:
    subprocess.run(["git", "add"] + files, check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)


def _score_model(adapter_path: str, data_path: str) -> float:
    """Run the tool-calling scorer. Deferred import so mlx is not required for unit tests."""
    from recipes.toolcalling.eval import evaluate as run_evaluate  # noqa: F401
    return run_evaluate(adapter_path, data_path)


def ratchet_loop(
    recipe_path: str,
    n_experiments: int = 10,
    target_score: float = 0.90,
    state_path: str = "loop_state.json",
    seed: int = 42,
) -> None:
    """
    Run the auto-search ratchet.

    Each iteration:
      1. Propose a single-param config change.
      2. Train with the proposed config.
      3. Score the resulting adapter.
      4. If score > best: update recipe.yaml + state + program.md, git commit.
         If score <= best: discard (adapter dir is overwritten next run).
      5. Repeat until n_experiments or target_score is reached.
    """
    recipe_path = Path(recipe_path)
    state_path = Path(state_path)
    program_md = recipe_path.parent / "program.md"
    scoring_data = recipe_path.parent / "data" / "valid.jsonl"

    rng = random.Random(seed)
    state = load_state(state_path)
    cfg = load_recipe(recipe_path)

    console.print(
        f"[bold]Starting ratchet loop[/] -- {n_experiments} experiments, target={target_score}"
    )
    console.print(f"Current best score: [cyan]{state.best_score:.4f}[/]")

    results: list[tuple[int, float, bool]] = []

    for _ in range(1, n_experiments + 1):
        exp_num = state.experiment + 1
        console.rule(f"Experiment {exp_num}")

        proposed = propose_config(cfg, SEARCH_SPACE, rng)
        proposed = dataclasses.replace(
            proposed,
            adapter_path=f"{cfg.adapter_path}_exp{exp_num:03d}",
        )

        console.print(
            f"Proposed: lr={proposed.learning_rate} rank={proposed.lora_rank} "
            f"layers={proposed.lora_layers} batch={proposed.batch_size}"
        )

        try:
            run_training(proposed)
            score = _score_model(proposed.adapter_path, str(scoring_data))
        except Exception as exc:
            console.print(f"[red]Experiment {exp_num} failed:[/] {exc}")
            results.append((exp_num, 0.0, False))
            state.experiment = exp_num
            save_state(state, state_path)
            continue

        kept = score > state.best_score
        results.append((exp_num, score, kept))

        if kept:
            console.print(f"[green]KEEP[/] {score:.4f} > {state.best_score:.4f}")
            state.best_score = score
            state.experiment = exp_num
            state.best_config = {
                "learning_rate": proposed.learning_rate,
                "lora_rank": proposed.lora_rank,
                "lora_layers": proposed.lora_layers,
                "batch_size": proposed.batch_size,
                "adapter_path": proposed.adapter_path,
            }

            recipe_data = yaml.safe_load(recipe_path.read_text())
            recipe_data["learning_rate"] = proposed.learning_rate
            recipe_data["lora_rank"] = proposed.lora_rank
            recipe_data["lora_layers"] = proposed.lora_layers
            recipe_data["batch_size"] = proposed.batch_size
            recipe_path.write_text(yaml.dump(recipe_data, default_flow_style=False))

            save_state(state, state_path)
            if program_md.exists():
                update_program_md(program_md, score, exp_num)

            commit_files = [str(recipe_path), str(state_path)]
            if program_md.exists():
                commit_files.append(str(program_md))
            _git_commit(
                commit_files,
                f"loop: exp {exp_num:03d} score={score:.4f} "
                f"lr={proposed.learning_rate} rank={proposed.lora_rank}",
            )
        else:
            console.print(f"[yellow]DISCARD[/] {score:.4f} <= {state.best_score:.4f}")
            state.experiment = exp_num
            save_state(state, state_path)

        if state.best_score >= target_score:
            console.print(f"[bold green]Target {target_score} reached -- stopping.[/]")
            break

    _print_summary(results, state.best_score)


def _print_summary(results: list[tuple[int, float, bool]], best: float) -> None:
    table = Table(title="Loop summary")
    table.add_column("Exp", style="dim")
    table.add_column("Score")
    table.add_column("Kept")
    for exp, score, kept in results:
        table.add_row(str(exp), f"{score:.4f}", "v" if kept else "x")
    console.print(table)
    console.print(f"Best score: [bold cyan]{best:.4f}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mlx-forge auto-search ratchet loop")
    parser.add_argument("--recipe", required=True, help="Path to recipe.yaml")
    parser.add_argument("--n-experiments", type=int, default=10)
    parser.add_argument("--target-score", type=float, default=0.90)
    parser.add_argument("--state-path", default="loop_state.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ratchet_loop(
        recipe_path=args.recipe,
        n_experiments=args.n_experiments,
        target_score=args.target_score,
        state_path=args.state_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
