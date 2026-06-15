# Changelog

All notable changes to mlx-forge are documented here.

## [0.1.0] — 2026-06-14

### Added
- Full fine-tuning pipeline: download → train (LoRA) → eval → fuse → export GGUF → push to HF
- Auto-search ratchet loop — propose one hyperparameter change, train, score, keep if better, repeat overnight
- Interactive Textual TUI (`mlx-forge`) with sidebar navigation, live output, and Ctrl+X cancel
- 3 task recipes ready to run: `toolcalling`, `edge_android`, `healthcare_coding`
- Data flywheel recipe — model generates its own training data, judges quality, retrains
- Data validation + format conversion (`mlx-forge data validate/convert`) for JSONL, CSV, Alpaca
- Transfer mode — run the search loop on a small sibling model, apply best config to a large ship model
- GitHub Pages landing page at `https://abdouloued.github.io/mlx-forge/`
- 120 unit tests, all passing without a model or GPU

### Architecture
- `core/` — generic machinery (train, eval, fuse, export, push, loop, transfer, datakit, TUI, CLI)
- `recipes/` — task-specific config + data + eval (pluggable pattern)
- `shared/formats/` — reusable format validators

### Constraints (by design)
- MLX only — no NVIDIA/CUDA
- LoRA/QLoRA only — no full fine-tuning of large models
- Publishing always manual — the loop never calls push or export automatically
- Base model must be HuggingFace safetensors — GGUF cannot be a fine-tuning starting point
