# Tool-calling Auto-Search Program

This file guides the auto-search loop (Phase 2). The loop reads this file
before proposing each experiment. Edit it to steer the search.

## Current best score
(updated by the loop after each kept experiment)
score: 0.0
experiment: baseline

## Search space
The loop may vary ONLY these parameters in recipe.yaml:
- learning_rate: try values in [5e-5, 1e-4, 2e-4, 5e-4]
- lora_rank: try values in [4, 8, 16, 32]
- lora_layers: try values in [8, 16, 24, 32]
- iters: must stay at 500 (fixed budget for comparability)
- batch_size: try values in [2, 4, 8]

## Constraints
- Do NOT change base_model.
- Do NOT change data_dir or the eval.
- Do NOT attempt full fine-tuning (no --no-adapter flag).
- Each experiment must use the same fixed iters budget.

## Notes for the loop
- Higher lora_rank = more expressive adapter but slower training.
- learning_rate too high → unstable loss. Too low → slow convergence.
- If the last N experiments all got worse, try a more conservative LR.
- A score above 0.85 is strong. Stop if you reach 0.90.
