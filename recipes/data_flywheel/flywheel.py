"""
Data-flywheel for mlx-forge.

Pattern: generate → judge → filter → append to training data → retrain.

Each round:
  1. Run the current model on a set of seed prompts to generate candidate responses.
  2. Score each candidate with the recipe's judge function.
  3. Keep candidates above the quality threshold.
  4. Append kept examples to train.jsonl as new supervised examples.
  5. Retrain on the expanded dataset.

The flywheel is recipe-agnostic: plug in any score_fn that maps
(response: str, seed: dict) -> float (0.0 to 1.0).

The default judge uses the toolcalling scorer so the flywheel can bootstrap
from the toolcalling recipe out of the box.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()

ScoreFn = Callable[[str, dict], float]


@dataclass
class FlywheelRound:
    round_n: int
    generated: int
    kept: int
    train_size_before: int
    train_size_after: int

    @property
    def kept_rate(self) -> float:
        return self.kept / self.generated if self.generated else 0.0


def judge_candidates(candidates: List[dict], threshold: float = 0.75) -> List[dict]:
    """Return only candidates whose score meets or exceeds threshold."""
    return [c for c in candidates if c["score"] >= threshold]


def format_as_training_example(seed: dict, response: str) -> dict:
    """
    Combine a seed prompt with a generated response into a supervised training example.

    seed must contain 'messages' (list) and optionally 'tools'.
    The assistant turn is appended as the final message.
    """
    messages = list(seed.get("messages", []))
    messages.append({"role": "assistant", "content": response})
    example: dict = {"messages": messages}
    if seed.get("tools"):
        example["tools"] = seed["tools"]
    return example


def append_to_training_data(examples: List[dict], train_path: Path | str) -> int:
    """Append examples to train.jsonl. Returns the number appended."""
    if not examples:
        return 0
    path = Path(train_path)
    with path.open("a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def _load_seeds(seeds_path: Path) -> List[dict]:
    return [json.loads(l) for l in seeds_path.read_text().splitlines() if l.strip()]


def _generate_and_score(
    model_path: str,
    seeds: List[dict],
    score_fn: ScoreFn,
    n_per_seed: int,
) -> List[dict]:
    """Run inference on seeds, score each response, return candidate dicts."""
    from mlx_lm import load, generate  # type: ignore

    model, tokenizer = load(model_path)
    candidates = []

    for seed in seeds:
        prompt_messages = seed.get("messages", [])
        tools = seed.get("tools") or None
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        for _ in range(n_per_seed):
            response = generate(
                model, tokenizer, prompt=prompt, max_tokens=256, verbose=False
            )
            score = score_fn(response, seed)
            candidates.append({"response": response, "seed": seed, "score": score})

    return candidates


def run_flywheel(
    model_path: str,
    seeds_path: str,
    train_path: str,
    score_fn: ScoreFn,
    n_rounds: int = 3,
    n_per_seed: int = 3,
    threshold: float = 0.75,
) -> List[FlywheelRound]:
    """
    Run the data-flywheel for n_rounds.

    Each round generates n_per_seed responses per seed, judges them,
    and appends the survivors to train_path. Does NOT trigger retraining —
    call core.train after each round (or after all rounds) as desired.

    Returns a list of FlywheelRound stats, one per round.
    """
    seeds = _load_seeds(Path(seeds_path))
    train = Path(train_path)
    rounds: List[FlywheelRound] = []

    for r in range(1, n_rounds + 1):
        console.rule(f"Flywheel round {r}/{n_rounds}")
        size_before = _count_lines(train)

        candidates = _generate_and_score(model_path, seeds, score_fn, n_per_seed)
        kept = judge_candidates(candidates, threshold)

        new_examples = [
            format_as_training_example(c["seed"], c["response"]) for c in kept
        ]
        append_to_training_data(new_examples, train)
        size_after = _count_lines(train)

        fr = FlywheelRound(
            round_n=r,
            generated=len(candidates),
            kept=len(kept),
            train_size_before=size_before,
            train_size_after=size_after,
        )
        rounds.append(fr)
        console.print(
            f"  generated={fr.generated}  kept={fr.kept}  "
            f"kept_rate={fr.kept_rate:.0%}  "
            f"train: {size_before} -> {size_after}"
        )

    _print_summary(rounds, train)
    return rounds


def _print_summary(rounds: List[FlywheelRound], train: Path) -> None:
    table = Table(title="Flywheel summary")
    table.add_column("Round", style="dim")
    table.add_column("Generated")
    table.add_column("Kept")
    table.add_column("Kept %")
    table.add_column("Train size")
    for r in rounds:
        table.add_row(
            str(r.round_n), str(r.generated), str(r.kept),
            f"{r.kept_rate:.0%}", str(r.train_size_after),
        )
    console.print(table)
    total_added = sum(r.kept for r in rounds)
    console.print(f"Total new training examples added: [bold cyan]{total_added}[/]")


def _default_score_fn(response: str, seed: dict) -> float:
    """Default judge: use toolcalling scorer if expected field present, else 1.0 for non-empty."""
    if "expected" in seed:
        from recipes.toolcalling.eval import score_response as tc_score  # type: ignore
        return tc_score(response, seed)
    return 1.0 if response.strip() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mlx-forge data flywheel")
    parser.add_argument("--model-path", required=True, help="Path to model or adapter")
    parser.add_argument("--seeds-path", required=True, help="JSONL of seed prompts")
    parser.add_argument(
        "--train-path",
        required=True,
        help="JSONL training file to append good examples to",
    )
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--n-per-seed", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    run_flywheel(
        model_path=args.model_path,
        seeds_path=args.seeds_path,
        train_path=args.train_path,
        score_fn=_default_score_fn,
        n_rounds=args.n_rounds,
        n_per_seed=args.n_per_seed,
        threshold=args.threshold,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
