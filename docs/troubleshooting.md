# Troubleshooting

Common problems and how to fix them. If your issue isn't here, [open an issue](https://github.com/abdouloued/mlx-forge/issues).

---

## 1. OOM (out of memory) during training

**Symptoms:** Training exits immediately with an out-of-memory error, or the Mac fan spins up and training stalls.

**Fix:** Reduce memory pressure in `recipe.yaml`:
```yaml
grad_checkpoint: true    # halves activation memory — enable first
batch_size: 2            # default is 4; 2 uses half the gradient memory
lora_rank: 4             # default is 8; lower rank = fewer parameters
lora_layers: 8           # default is 16; fewer layers = less memory
```
Apply all four changes, then re-run. Once training completes, tune back up one setting at a time.

---

## 2. "Training failed with exit code 1"

**Symptoms:** The TUI shows "Training failed with exit code 1" or the CLI raises `subprocess.CalledProcessError`.

**Steps:**
1. Run the training command manually to see the raw error:
   ```bash
   uv run python -m mlx_lm lora --help
   ```
2. Common causes:
   - **mlx-lm version mismatch:** Run `uv sync` to update to the pinned version.
   - **Data path wrong:** Check that `data_dir` in `recipe.yaml` points to a directory containing `train.jsonl` and `valid.jsonl`.
   - **Base model not downloaded:** Run `uv run python -c "from mlx_lm import load; load('your-model-id')"` to trigger download.

---

## 3. Eval score stays at 0.0 after training

**Symptoms:** Training completes but `evaluate()` returns 0.0.

**Checklist:**
1. Validate your data format:
   ```bash
   uv run python -m core.datakit validate recipes/your_recipe/data/valid.jsonl
   ```
   Zero errors required. Even one malformed example can cause the scorer to skip all examples.

2. Check the adapter path exists:
   ```bash
   ls adapters/your_recipe/
   # Should contain: adapter_config.json, adapters.safetensors, etc.
   ```

3. Check the recipe name matches the directory:
   ```bash
   # The recipe is loaded by directory name, not by recipe.yaml content
   ls recipes/  # names must match what you pass to --recipe
   ```

4. Run `eval.py` manually with verbose output to see what's failing:
   ```bash
   uv run python -c "
   from recipes.toolcalling.eval import evaluate
   score = evaluate('mlx-community/Qwen2.5-7B-Instruct-4bit',
                    'recipes/toolcalling/data/valid.jsonl',
                    adapter_path='adapters/toolcalling')
   print('score:', score)
   "
   ```

---

## 4. GGUF export fails

**Symptoms:** `core/export_gguf.py` raises `FileNotFoundError` or `subprocess.CalledProcessError`.

**Cause:** `llama-quantize` is not on `$PATH`.

**Fix:**
```bash
# Check if binary exists:
which llama-quantize || which llama-quantize-metal || which llama-cli

# If not found, build llama.cpp:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=on
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)

# Add to PATH:
export PATH="$PATH:$(pwd)/build/bin"
# Add that export to your ~/.zshrc or ~/.bashrc
```

Note: the binary name changes between llama.cpp releases (`llama-quantize`, `llama-quantize-metal`, `quantize`). Check `build/bin/` for the actual binary name.

---

## 5. HuggingFace push fails with 401 / 403

**Symptoms:** `push_hf.py` raises `huggingface_hub.errors.RepositoryNotFoundError` or returns HTTP 401.

**Fix:**
```bash
uv run huggingface-cli login
# Paste a token from https://huggingface.co/settings/tokens
# Token must have "write" permission
```

For private repos, the token needs the `read` scope on the repo as well.

---

## 6. TUI won't launch

**Symptoms:** `mlx-forge` or `uv run python -m core.tui` exits immediately or shows a blank screen.

**Check Textual version:**
```bash
uv run python -c "import textual; print(textual.__version__)"
# Requires 0.70+
```

**Fix:**
```bash
uv sync
```

**Terminal compatibility:** Textual requires a terminal that supports ANSI escape codes. iTerm2, Warp, and the macOS built-in Terminal all work. Avoid running through SSH without `TERM=xterm-256color` set.

---

## 7. Loop runs but score never improves

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Dataset too small | Ensure ≥100 train examples and ≥30 valid examples per recipe |
| Search space exhausted | Add new values to `SEARCH_SPACE` in `core/loop.py` |
| Eval is non-deterministic | Set `temperature=0.0` in the `generate()` call in `eval.py` |
| `iters` budget too short | Increase `iters` in `recipe.yaml` (at least 500 for 7B models) |
| Wrong base model | Verify the model is fine-tunable (instruction-tuned variants often work better as starting points) |

The loop writes progress to `loop_state.json`. Inspect it to see the score history:
```bash
cat loop_state.json | python3 -m json.tool | grep score
```

---

## 8. Data validation errors

**Symptoms:** `uv run python -m core.datakit validate` reports errors.

**Common error messages and fixes:**

| Error | Fix |
|-------|-----|
| `missing 'messages' key` | Each line must be `{"messages": [...]}` |
| `invalid role 'Human'` | Roles must be `system`, `user`, `assistant`, or `tool` |
| `empty content` | Remove examples where any message has `"content": ""` |
| `not valid JSON` | Run `python3 -m json.tool < train.jsonl` to find the bad line |
| `conversation must start with system or user` | First message cannot be from `assistant` |

---

## 9. `mlx_lm` import error

**Symptoms:** `ModuleNotFoundError: No module named 'mlx_lm'`

**Fix:**
```bash
uv sync
# or:
uv add mlx-lm
```

mlx-lm requires Apple Silicon. It will not install on Intel Macs or Linux/Windows.

---

## 10. Git commit fails inside the loop

**Symptoms:** The loop runs but does not commit winning configs to git.

**Cause:** The project directory is not a git repository.

**Fix:**
```bash
git init
git add .
git commit -m "initial commit"
```

The loop silently skips git operations if `git` is not available or the directory is not a git repo. This is intentional — the loop still works, but you lose the experiment log.
