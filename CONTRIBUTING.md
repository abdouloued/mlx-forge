# Contributing to mlx-forge

mlx-forge exists because we believe Apple Silicon is purpose-built for the next wave of AI: local inference, open-weight models, privacy-preserving fine-tuning that runs on the device in your pocket. If you share that belief, welcome.

---

## Philosophy

Three things that never change, no matter the contribution:

1. **Evals first.** A fine-tuned model without a trustworthy eval is just a loss curve. Every recipe ships a scorer that returns a single comparable 0–1 score. The loop only keeps experiments that beat the current best.

2. **Apple Silicon only.** MLX, not CUDA. This is not a limitation — it is the point. Unified memory changes the economics of local inference completely.

3. **Publishing is always manual.** The loop never calls `push_hf` or `export_gguf`. The human decides when a model is ready to ship. This is a design constraint, not an oversight.

---

## Quick dev setup

```bash
git clone https://github.com/abdouloued/mlx-forge
cd mlx-forge
uv sync --group dev

# Run all tests (no model or network required)
uv run pytest -v

# Run the linter
uv run ruff check .
```

All 120 tests run in ~0.2s with no model download. If a test requires a real model, it belongs in a separate integration test directory and is skipped by default.

---

## Adding a new recipe

A recipe is a self-contained fine-tuning task: config, data, and a task-specific scorer. The `toolcalling/` recipe is the reference — study it first.

### Step 1: Copy the template

```bash
cp -r recipes/toolcalling recipes/my_recipe
```

### Step 2: Edit `recipe.yaml`

```yaml
base_model: mlx-community/Qwen2.5-7B-Instruct-4bit  # must be HF safetensors
mode: direct

lora_rank: 8        # 4–32; start at 8
lora_alpha: 16.0    # typically 2× rank
lora_layers: 16     # how many transformer layers to fine-tune
iters: 500          # training iterations; keep fixed across loop experiments
learning_rate: 0.0001
batch_size: 4
grad_checkpoint: true

data_dir: recipes/my_recipe/data
adapter_path: adapters/my_recipe
fused_path: fused/my_recipe
```

**Field reference:**

| Field | Effect |
|-------|--------|
| `lora_rank` | LoRA decomposition rank. Higher = more parameters, more capacity, slower. |
| `lora_layers` | Number of transformer layers to apply LoRA to (from the output layer back). |
| `iters` | Keep this fixed across all loop experiments so scores are comparable. |
| `grad_checkpoint` | Set `true` on M1/M2 with 16GB RAM to avoid OOM. Slight speed penalty. |

### Step 3: Prepare your data

See [docs/data-guide.md](docs/data-guide.md) for the full format specification.

**Minimum requirements:**
- `data/train.jsonl` — at least **100 examples** (200+ recommended)
- `data/valid.jsonl` — at least **30 examples** (50+ recommended)

Each line is a chat JSONL object:
```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Validate your data before submitting:
```bash
uv run mlx-forge data validate recipes/my_recipe/data/train.jsonl
```

### Step 4: Implement `eval.py`

This is the only file you **must** write from scratch. It must expose:

```python
def evaluate(model_path: str, data_path: str, adapter_path: Optional[str] = None) -> float:
    """
    Run the model on validation examples and return the mean score (0–1).
    Higher is better. The loop uses this as the ratchet.
    """
```

The score must be:
- **Deterministic** given the same model + data (or close enough)
- **Comparable across runs** — the loop relies on `score > best_score` comparisons
- **Between 0.0 and 1.0**

**Example scorer patterns:**

| Task type | Scoring approach |
|-----------|-----------------|
| JSON/tool calling | Parse output as JSON, check keys match expected |
| Classification | Exact match on label |
| Extraction | Check if extracted value appears in response |
| Short-form QA | Keyword presence + length budget |
| Refusal tasks | Detect refusal phrases for out-of-scope inputs |
| Code generation | Run the generated code, check output |

### Step 5: Write `program.md`

This file guides the auto-search loop. It should describe:
- What the model should accomplish (task description)
- Which hyperparameters should NOT be changed (e.g., base_model)
- The quality bar (what score means "done"?)
- Any domain-specific constraints

### Step 6: Add tests

Mirror the pattern in `tests/test_eval.py`:

```python
# tests/test_my_recipe_eval.py
from recipes.my_recipe.eval import score_response

def test_correct_output():
    assert score_response("correct output", {"expected": "..."}) == 1.0

def test_wrong_output():
    assert score_response("wrong output", {"expected": "..."}) == 0.0

def test_empty_output():
    assert score_response("", {"expected": "..."}) == 0.0
```

Tests must pass without a model (`uv run pytest tests/test_my_recipe_eval.py -v`).

---

## Modifying core code

The `core/` directory is recipe-agnostic. If you change it, all recipes are affected.

**Rules:**
- Do not add recipe-specific logic to `core/`
- Keep `RecipeConfig` in `core/config.py` — it is the contract between core and recipes
- The `evaluate()` interface (`model_path, data_path, adapter_path=None → float`) must remain stable

**Before opening a PR that touches core:**
1. Run `uv run pytest -v` — all 120 tests must pass
2. Run `uv run ruff check .` — zero lint errors
3. Test manually with at least the `toolcalling` recipe

---

## PR guidelines

**Commit format:**
```
type(scope): short description

feat(recipes): add SQL generation recipe
fix(loop): handle missing valid.jsonl gracefully
docs: add troubleshooting guide for OOM errors
test(datakit): add CSV conversion edge case
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

**PR checklist:**
- [ ] Tests pass: `uv run pytest -v`
- [ ] Lint passes: `uv run ruff check .`
- [ ] No model files committed (adapters/, fused/, exports/, *.gguf, *.safetensors)
- [ ] If adding a recipe: eval.py has `evaluate(model_path, data_path, adapter_path=None)`
- [ ] If adding a recipe: ≥100 train examples, ≥30 valid examples, data validates cleanly

---

## Hard constraints (never negotiate these)

| Constraint | Reason |
|-----------|--------|
| MLX only | Apple Silicon unified memory is the product's point of difference |
| LoRA/QLoRA only | Full fine-tuning of 7B+ is impractical on consumer hardware |
| No automatic publishing | Users must control when their models ship |
| Base model must be HF safetensors | GGUF cannot be used as a fine-tuning starting point |
| Healthcare data must remain synthetic | Real PHI/HIPAA data cannot be in a public repo |

---

## Questions?

Open an issue with the `question` label or start a GitHub Discussion. We read everything.
