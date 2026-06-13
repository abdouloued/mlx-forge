"""
Tool-calling eval for mlx-forge.

Scoring rubric (each worth 0.25 of the total score):
  1. called_tool   — model output is parseable as a tool call
  2. valid_json    — arguments parse as valid JSON dict
  3. correct_name  — function name matches expected
  4. correct_args  — all keys in expected["arguments"] are present in the response

Entry points:
  score_response(response, example) -> float   (pure, unit-testable)
  evaluate(model_path, data_path)   -> float   (runs inference + scoring)
  main()                            -> CLI
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def parse_tool_call(response: str) -> Optional[dict]:
    """Try to parse `response` as a {name, arguments} tool call. Returns None on failure."""
    response = response.strip()
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "name" not in data or not isinstance(data["name"], str):
        return None
    if "arguments" not in data or not isinstance(data["arguments"], dict):
        return None
    return data


def score_response(response: str, example: dict) -> float:
    """
    Score a single model response against an example's expected tool call.

    Rubric (each criterion = 0.25):
      called_tool  — response is a parseable tool call, not prose
      valid_json   — arguments is a dict (included in parse check)
      correct_name — function name matches expected
      correct_args — all keys in expected["arguments"] are present in response args
    """
    expected = example.get("expected", {})
    expected_name: str = expected.get("name", "")
    expected_args: dict = expected.get("arguments", {})

    parsed = parse_tool_call(response)

    called_tool = parsed is not None
    valid_json = called_tool
    correct_name = called_tool and parsed["name"] == expected_name
    correct_args = (
        called_tool
        and all(
            k in parsed["arguments"] and parsed["arguments"][k] == v
            for k, v in expected_args.items()
        )
    )

    return sum([called_tool, valid_json, correct_name, correct_args]) / 4.0


def evaluate(model_path: str, data_path: str) -> float:
    """
    Run the model on validation examples and return the mean score (0-1).

    Requires mlx-lm and Apple Silicon. Imports are deferred so unit tests
    can import this module without mlx installed.
    """
    from mlx_lm import load, generate  # type: ignore

    model, tokenizer = load(model_path)
    examples = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]

    scores = []
    for ex in examples:
        messages = ex["messages"]
        # valid.jsonl: last message is the expected assistant turn — exclude it
        prompt_messages = messages[:-1]
        tools = ex.get("tools", [])

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False,
        )

        scores.append(score_response(response, ex))

    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool-calling fine-tune")
    parser.add_argument("--model-path", required=True, help="Path to fused model or adapter")
    parser.add_argument(
        "--data-path",
        default="recipes/toolcalling/data/valid.jsonl",
        help="Path to validation JSONL",
    )
    args = parser.parse_args()
    score = evaluate(args.model_path, args.data_path)
    print(f"tool_calling_score={score:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
