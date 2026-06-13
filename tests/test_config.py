import pytest
from pathlib import Path
from core.config import RecipeConfig, load_recipe


def test_defaults():
    cfg = RecipeConfig(base_model="mlx-community/Qwen2.5-7B-Instruct-4bit")
    assert cfg.mode == "direct"
    assert cfg.lora_rank == 8
    assert cfg.iters == 1000


def test_transfer_mode_requires_loop_model():
    with pytest.raises(ValueError, match="loop_model"):
        RecipeConfig(
            base_model="mlx-community/Qwen2.5-32B-Instruct-4bit",
            mode="transfer",
        ).validate()


def test_transfer_mode_valid():
    cfg = RecipeConfig(
        base_model="mlx-community/Qwen2.5-32B-Instruct-4bit",
        mode="transfer",
        loop_model="mlx-community/Phi-4-mini-instruct-4bit",
    )
    cfg.validate()  # should not raise


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        RecipeConfig(base_model="any/model", mode="full").validate()


def test_load_recipe(tmp_path):
    yaml_content = """
base_model: mlx-community/Qwen2.5-7B-Instruct-4bit
mode: direct
lora_rank: 16
iters: 500
data_dir: data
adapter_path: adapters
"""
    recipe_file = tmp_path / "recipe.yaml"
    recipe_file.write_text(yaml_content)
    cfg = load_recipe(recipe_file)
    assert cfg.lora_rank == 16
    assert cfg.iters == 500
    assert cfg.mode == "direct"
