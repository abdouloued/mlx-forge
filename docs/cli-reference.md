# CLI Reference

All commands are run via `uv run python -m <module>`. Replace `recipes/toolcalling/recipe.yaml` with your recipe path.

---

## Training

```bash
uv run python -m core.train --recipe recipes/toolcalling/recipe.yaml
```

Runs `mlx_lm lora` with the hyperparameters from `recipe.yaml`. Adapter weights are saved to `adapter_path` (default: `adapters/toolcalling/`).

**Options:**
| Flag | Description |
|------|-------------|
| `--recipe PATH` | Path to `recipe.yaml` (required) |

---

## Evaluation

```bash
uv run python -m recipes.toolcalling.eval \
  --model-path mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data-path recipes/toolcalling/data/valid.jsonl \
  --adapter-path adapters/toolcalling
```

Runs the recipe-specific scorer on the validation set and prints the mean score.

**Expected output:**
```
tool_calling_score=0.8750
```

---

## Fuse

```bash
uv run python -m core.fuse --recipe recipes/toolcalling/recipe.yaml
```

Merges the LoRA adapter into the base model weights. Output saved to `fused_path` (default: `fused/toolcalling/`).

---

## Auto-search loop

```bash
uv run python -m core.loop \
  --recipe recipes/toolcalling/recipe.yaml \
  --n-experiments 20 \
  --target-score 0.90 \
  --seed 42
```

Runs the ratchet loop: propose → train → score → keep if better → repeat.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--recipe PATH` | — | Path to `recipe.yaml` (required) |
| `--n-experiments N` | 10 | Maximum number of experiments |
| `--target-score FLOAT` | 1.0 | Stop early if score reaches this threshold |
| `--seed INT` | None | Random seed for reproducibility |

**Loop state:** Saved to `loop_state.json` in the working directory. Contains best score, best config, and all experiment results.

---

## Export to GGUF

```bash
uv run python -m core.export_gguf \
  --fused-path fused/toolcalling \
  --output-gguf exports/toolcalling/model-q4_k_m.gguf \
  --llama-cpp-dir ~/llama.cpp \
  --quantization Q4_K_M \
  --system-prompt "You are a helpful assistant with access to tools."
```

Converts fused safetensors weights to GGUF format and generates an Ollama `Modelfile`.

**Options:**
| Flag | Description |
|------|-------------|
| `--fused-path PATH` | Path to fused model weights (required) |
| `--output-gguf PATH` | Output GGUF file path (required) |
| `--llama-cpp-dir DIR` | Path to llama.cpp directory (required) |
| `--quantization TYPE` | GGUF quantisation type (default: `Q4_K_M`) |
| `--system-prompt TEXT` | System prompt for the Ollama Modelfile |

**After export:**
```bash
cd exports/toolcalling
ollama create my-model -f Modelfile
ollama run my-model
```

---

## Push to Hugging Face

```bash
uv run python -m core.push_hf \
  --recipe recipes/toolcalling/recipe.yaml \
  --repo-id your-username/qwen2-5-7b-toolcalling
```

Uploads the fused model to Hugging Face Hub.

**Options:**
| Flag | Description |
|------|-------------|
| `--recipe PATH` | Path to `recipe.yaml` (required) |
| `--repo-id REPO` | HF repo in `owner/name` format (required) |
| `--public` | Make the repo public (default: private) |

**Authenticate first:**
```bash
uv run huggingface-cli login
```

---

## Data validation

```bash
uv run python -m core.datakit validate recipes/toolcalling/data/train.jsonl
```

**Expected output:**
```
Total lines: 100 │ Valid: 100 │ Errors: 0
```

---

## Data conversion

**From CSV:**
```bash
uv run python -m core.datakit convert \
  --from csv \
  --input my_data.csv \
  --input-col question \
  --output-col answer \
  --system "You are a helpful assistant." \
  --output recipes/my_recipe/data/train.jsonl
```

**From Q&A JSONL (input/output fields):**
```bash
uv run python -m core.datakit convert \
  --from qa \
  --input qa_pairs.jsonl \
  --output recipes/my_recipe/data/train.jsonl
```

**From Alpaca instruction format:**
```bash
uv run python -m core.datakit convert \
  --from instruction \
  --input alpaca.jsonl \
  --output recipes/my_recipe/data/train.jsonl
```

---

## Tests

```bash
uv run pytest -v
```

Runs all 120 unit tests. No model download or Apple Silicon required. Completes in ~0.2s.

```bash
uv run pytest tests/test_eval.py -v          # toolcalling scorer only
uv run pytest tests/test_healthcare_eval.py  # healthcare scorer only
uv run pytest -k "fuse" -v                   # all fuse-related tests
```

---

## Linter

```bash
uv run ruff check .
uv run ruff check . --fix    # auto-fix safe issues
```

Zero errors required before opening a PR.

---

## Interactive TUI

```bash
uv run mlx-forge
# or:
uv run python -m core.tui
```

Opens the Textual TUI. Navigate with arrow keys, fill fields, press Enter to run actions.
Use `ctrl+x` to cancel a running operation.
