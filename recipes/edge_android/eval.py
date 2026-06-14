"""eval.py — standard evaluate() interface for mlx-forge cli/loop."""
from __future__ import annotations
from typing import Optional


def evaluate(model_path: str, data_path: str, adapter_path: Optional[str] = None) -> float:
    from mlx_lm import load, generate  # type: ignore
    import json
    from pathlib import Path
    from recipes.edge_android.scorer import score_response

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    examples = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]
    scores = []
    for ex in examples:
        messages = ex["messages"][:-1]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
        scores.append(score_response(response, ex))
    return sum(scores) / len(scores) if scores else 0.0
