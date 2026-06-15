# mlx-forge Architecture

A top-to-bottom guide to how the pieces fit together. Read this before touching core/.

---

## Directory Layout

```
mlx-forge/
├── core/           # Recipe-agnostic engine — train, fuse, eval orchestration
├── recipes/        # Task-specific plugins (data + eval + config)
│   ├── toolcalling/
│   ├── edge_android/
│   ├── healthcare_coding/
│   └── data_flywheel/
├── shared/         # Cross-recipe utilities (tool schema, format helpers)
├── tests/          # Pytest suite — runs without any model download
├── docs/           # This file + data-guide.md + troubleshooting.md
└── index.html      # GitHub Pages landing page
```

---

## Data Flow: recipe.yaml → trained adapter

```
recipe.yaml
    │
    ▼
load_recipe()          core/config.py        → RecipeConfig dataclass
    │
    ├── direct mode ──────────────────────────────────────────────────┐
    │                                                                 │
    ▼                                                                 │
run_training(cfg)      core/train.py         → mlx_lm lora (subprocess)
    │
    ▼
build_fuse_command(cfg) core/fuse.py         → mlx_lm fuse (subprocess)
    │
    ▼
mod.evaluate(...)      recipes/{name}/eval.py → float score (0–1)
    │
    └── transfer mode ────────────────────────────────────────────────┘
        (loop runs on cfg.loop_model, best config applied to cfg.base_model)
```

---

## The Ratchet Loop (`core/loop.py`)

The loop implements one-param-at-a-time hill climbing over `SEARCH_SPACE`:

```
SEARCH_SPACE = {
    "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
    "lora_rank":     [4, 8, 16, 32],
    "lora_layers":   [8, 16, 32],
    "batch_size":    [2, 4, 8],
}
```

Each iteration:
1. Pick a random param from `SEARCH_SPACE`.
2. Pick a random value for it (different from current).
3. Train with `run_training(proposed)`.
4. Score with `mod.evaluate(base_model, data_path, adapter_path=adapter_path)`.
5. If `score > best_score`: keep proposed config, save to `loop_state.json`.
6. If `score <= best_score`: discard, keep current best.
7. Commit the winner's YAML to git (if in a repo).

The loop never touches `eval.py`, `core/`, or training data — only `recipe.yaml` fields.

### Transfer mode

When `mode: transfer` is set in `recipe.yaml`, the loop trains on `cfg.loop_model`
(a small, fast model) to find the best hyperparameters cheaply. Once the loop
terminates, the user applies the winning config to `cfg.base_model` (the full ship model)
with a single `mlx-forge train` call.

Use transfer mode when your base model is 7B+ and loop iterations are slow.

---

## Recipe Plugin Interface

A recipe is a directory under `recipes/` that satisfies this interface:

```
recipes/my_recipe/
├── recipe.yaml         # RecipeConfig fields (see core/config.py)
├── data/
│   ├── train.jsonl     # ≥100 chat JSONL examples (see docs/data-guide.md)
│   └── valid.jsonl     # ≥30 examples used for scoring
├── eval.py             # MUST expose evaluate(model_path, data_path, adapter_path=None) → float
├── scorer.py           # Low-level scoring logic (called by eval.py)
└── program.md          # Loop search constraints (optional but recommended)
```

### Dynamic import

`core/loop.py` and `core/cli.py` both import eval modules dynamically:

```python
import importlib
mod = importlib.import_module(f"recipes.{recipe_name}.eval")
score = mod.evaluate(cfg.base_model, str(data_path), adapter_path=cfg.adapter_path)
```

This means adding a recipe requires no changes to core code. The recipe name is
derived from the recipe directory name (`recipe_path.parent.name`).

### evaluate() contract

```python
def evaluate(
    model_path: str,      # path to base model (HF safetensors)
    data_path: str,       # path to valid.jsonl
    adapter_path: Optional[str] = None,  # path to trained LoRA adapter
) -> float:              # 0.0 – 1.0, higher is better
```

The score must be:
- **Deterministic** (or near-deterministic) given the same model + data
- **Comparable across runs** — the ratchet relies on `score > best_score`
- **Between 0.0 and 1.0**

---

## TUI Thread Safety (`core/tui.py`)

The TUI runs Textual on the main thread. All long-running operations (training,
eval, fuse, export, push) run in background threads via `@work(thread=True)`.

### The `_done()` pattern

Every `@work` method must call `self._done()` before returning — including early
returns on validation failure. Missing `_done()` causes the UI to freeze because
the spinner never stops.

```python
@work(thread=True)
def _run_train(self) -> None:
    cfg_path = self.query_one("#recipe-path", Input).value.strip()
    if not cfg_path:
        self.call_from_thread(self._out, "[red]No recipe path.[/]")
        self._done()   # ← required on EVERY exit path
        return
    # ... actual work ...
    self._done()       # ← required at end
```

### `call_from_thread`

Writing to Textual widgets from a background thread requires `call_from_thread`:

```python
self.call_from_thread(self._out, "[green]Done.[/]")
```

Direct widget writes from background threads cause race conditions.

### Process cancellation

`_stream()` stores the running `subprocess.Popen` handle in `self._proc`.
`action_cancel()` (bound to `ctrl+x`) calls `proc.terminate()` to send SIGTERM.

---

## mlx-lm CLI (0.31+)

mlx-lm 0.31 changed the CLI from module-style to subcommand-style:

| Old (≤0.30) | New (≥0.31) |
|-------------|-------------|
| `python -m mlx_lm.lora` | `python -m mlx_lm lora` |
| `python -m mlx_lm.fuse` | `python -m mlx_lm fuse` |
| `--rank N` CLI flag | `lora_parameters.rank` in YAML via `-c` |
| `--lora-layers N` CLI flag | `--num-layers N` |

`core/train.py` writes a temporary YAML for the LoRA parameters and passes it
with `-c temp.yaml`. The temp file is cleaned up after the subprocess exits.

---

## Testing

Tests live in `tests/` and run in ~0.2s with no model download:

```bash
uv run pytest -v
```

All model calls are avoided by testing scorers and config parsers directly.
Any test that would require a real model download belongs in `tests/integration/`
and is skipped by default (marked with `@pytest.mark.integration`).

Test naming convention: `tests/test_{module_name}.py` for core, `tests/test_{recipe_name}_eval.py` for recipes.

---

## Publishing (always manual)

The TUI's "Push to HuggingFace" button calls `core/push_hf.py`, which uses
`huggingface_hub.upload_folder`. The loop **never** calls push functions.
The user decides when a model is ready to ship.

GGUF export calls `llama-quantize` from llama.cpp. This binary must be on
`$PATH` — mlx-forge does not bundle it.
