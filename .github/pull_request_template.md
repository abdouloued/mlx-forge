## What does this PR do?

<!-- One or two sentences. -->

## Why?

<!-- The problem it solves or the need it addresses. -->

## Changes

- 
- 

## Testing

```bash
uv run pytest -v
# paste relevant output
```

## Checklist

- [ ] Tests pass (`uv run pytest -v`)
- [ ] Lint passes (`uv run ruff check .`)
- [ ] If adding a recipe: `eval.py` implements `evaluate(model_path, data_path, adapter_path=None) -> float`
- [ ] If adding a recipe: `data/train.jsonl` has ≥100 examples, `data/valid.jsonl` has ≥30 examples
- [ ] If adding a recipe: data validates cleanly (`mlx-forge data validate recipes/<name>/data/train.jsonl`)
- [ ] No model files committed (adapters/, fused/, exports/, *.gguf)
