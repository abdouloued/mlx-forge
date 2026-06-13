"""
Edge-android scoring module for mlx-forge.

Scoring rubric (two criteria, equal weight):
  1. correct (0.5) — response contains at least one expected keyword (case-insensitive)
  2. concise (0.5) — word count is within max_words budget

The conciseness criterion enforces a deployment constraint: edge/mobile assistants must
give short, direct answers to fit tight UX and latency budgets.

Entry points:
  parse_answer(response, keywords) -> bool   (pure, testable)
  score_response(response, example) -> float (pure, testable)
  run(model_path, data_path)        -> float (requires mlx-lm)
  main()                            -> CLI
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List


def parse_answer(response: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in response (case-insensitive)."""
    lower = response.lower()
    return any(kw.lower() in lower for kw in keywords)


def score_response(response: str, example: dict) -> float:
    """
    Score one response against its example.

    example keys:
      expected_keywords: list[str] — at least one must appear in the response
      max_words: int — response must be at or under this word count
    """
    if not response.strip():
        return 0.0

    keywords: List[str] = example.get("expected_keywords", [])
    max_words: int = example.get("max_words", 50)

    correct = parse_answer(response, keywords)
    word_count = len(response.split())
    concise = word_count <= max_words

    return (float(correct) + float(concise)) / 2.0


def run(model_path: str, data_path: str) -> float:
    """Run the model on validation examples and return the mean score (0-1)."""
    from mlx_lm import load, generate  # type: ignore

    model, tokenizer = load(model_path)
    examples = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]

    scores = []
    for ex in examples:
        messages = ex["messages"][:-1]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
        scores.append(score_response(response, ex))

    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Score edge-android fine-tune")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", default="recipes/edge_android/data/valid.jsonl")
    args = parser.parse_args()
    score = run(args.model_path, args.data_path)
    print(f"edge_android_score={score:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
