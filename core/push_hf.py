from __future__ import annotations
from pathlib import Path
from rich.console import Console
from huggingface_hub import upload_folder, create_repo

console = Console()


def build_repo_card(repo_id: str, base_model: str, recipe: str) -> str:
    return f"""---
base_model: {base_model}
library_name: mlx
tags:
  - mlx
  - lora
  - {recipe}
  - mlx-forge
---

# {repo_id}

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) using
[mlx-forge](https://github.com/your-org/mlx-forge) — the Mac-native fine-tuning factory.

**Recipe:** `{recipe}`

## Usage

```python
from mlx_lm import load, generate
model, tokenizer = load("{repo_id}")
response = generate(model, tokenizer, prompt="...", max_tokens=512)
print(response)
```

## Training

Fine-tuned with LoRA on Apple Silicon using MLX. See `adapter_config.json` for hyperparameters.
"""


def push_to_hf(
    fused_path: str,
    repo_id: str,
    base_model: str,
    recipe: str,
    private: bool = True,
) -> str:
    """Upload a fused model directory to Hugging Face Hub. Returns the repo URL."""
    console.print(f"[bold magenta]Creating repo:[/] {repo_id} (private={private})")
    create_repo(repo_id=repo_id, private=private, exist_ok=True)

    readme_path = Path(fused_path) / "README.md"
    readme_path.write_text(build_repo_card(repo_id, base_model, recipe))

    console.print(f"[bold magenta]Uploading:[/] {fused_path} → {repo_id}")
    upload_folder(
        folder_path=fused_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload mlx-forge fine-tune ({recipe})",
    )
    url = f"https://huggingface.co/{repo_id}"
    console.print(f"[bold magenta]Published:[/] {url}")
    return url


def main() -> None:
    import argparse
    from core.config import load_recipe
    parser = argparse.ArgumentParser(description="Push fused model to Hugging Face Hub")
    parser.add_argument("--recipe", required=True, help="Path to recipe.yaml")
    parser.add_argument("--repo-id", required=True, help="HF repo, e.g. myuser/my-model")
    parser.add_argument("--public", action="store_true", help="Make repo public (default: private)")
    args = parser.parse_args()
    cfg = load_recipe(args.recipe)
    push_to_hf(
        fused_path=cfg.fused_path,
        repo_id=args.repo_id,
        base_model=cfg.base_model,
        recipe=str(args.recipe),
        private=not args.public,
    )


if __name__ == "__main__":
    main()
