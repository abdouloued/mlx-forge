# Edge-Android Auto-Search Program

## Current best score
score: 0.0
experiment: baseline

## Deployment target
The model must be usable on a mid-range Android device:
- Quantized to Q4_K_M GGUF: ≤ 2GB on-disk
- Response latency: direct answers in ≤ 2 seconds on device

## Scoring
- 0.5 points: response contains the expected keyword(s)
- 0.5 points: response is within the max_words budget for that example
- Target score: 0.85+

## Search space
The loop may vary ONLY these parameters:
- learning_rate: [5e-5, 1e-4, 2e-4]
- lora_rank: [4, 8]         (keep small — larger rank → larger adapter → slower on device)
- lora_layers: [4, 8, 12]
- batch_size: [2, 4]
- iters: must stay at 300 (fixed budget)

## Constraints
- Do NOT increase lora_rank above 8 (adapter size budget)
- Do NOT change base_model (Phi-4-mini is the ship model)
- Do NOT change data_dir or the scorer
