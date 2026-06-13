# Data-Flywheel Program

## What this does
Uses the model itself to generate new training data. Good generations
(score >= threshold) are added to train.jsonl. The model is retrained
on the expanded dataset. Repeat.

## Current state
rounds_completed: 0
total_examples_added: 0
current_train_size: 20

## How to run one flywheel cycle

```bash
# Step 1: generate + filter + append
uv run python -m recipes.data_flywheel.flywheel \
  --model-path adapters/toolcalling \
  --seeds-path recipes/data_flywheel/seeds.jsonl \
  --train-path recipes/toolcalling/data/train.jsonl \
  --n-rounds 3 \
  --n-per-seed 3 \
  --threshold 0.75

# Step 2: retrain on expanded dataset
uv run python -m core.train --recipe recipes/data_flywheel/recipe.yaml
```

## Quality threshold guidance
- 0.75 (default): keeps responses that call the right tool with most args correct
- 0.90: strict — only near-perfect responses become training data
- 0.50: permissive — good for warming up a weak starting model

## When to stop
Stop adding rounds when the model's validation score on valid.jsonl stops improving
after retraining. Typically 3-5 rounds is sufficient for the toolcalling task.

## Important
- Review appended data periodically — the model can generate plausible-but-wrong examples
- Never lower threshold below 0.50 (garbage-in, garbage-out)
- Keep seeds.jsonl diverse — narrow seeds produce narrow training data
