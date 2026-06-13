"""Tests for transfer mode: apply best loop config from small model to ship model."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.config import RecipeConfig
from core.transfer import build_ship_config, apply_to_ship_model
from core.loop import LoopState, save_state


def make_recipe_yaml(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base_model: mlx-community/Qwen2.5-32B-Instruct-4bit\n"
        "mode: transfer\n"
        "loop_model: mlx-community/Phi-4-mini-instruct-4bit\n"
        "lora_rank: 8\n"
        "lora_alpha: 16.0\n"
        "lora_layers: 16\n"
        "iters: 500\n"
        "learning_rate: 0.0001\n"
        "batch_size: 4\n"
        "grad_checkpoint: true\n"
        "data_dir: data\n"
        "adapter_path: adapters/ship\n"
        "fused_path: fused/ship\n"
    )
    return recipe


def make_state(tmp_path: Path, best_config: dict) -> Path:
    state = LoopState(best_score=0.82, experiment=5, best_config=best_config)
    path = tmp_path / "loop_state.json"
    save_state(state, path)
    return path


# --- build_ship_config ---

def test_build_ship_config_uses_base_model(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = make_state(tmp_path, {"lora_rank": 16, "learning_rate": 2e-4,
                                        "lora_layers": 24, "batch_size": 4})
    cfg = build_ship_config(str(recipe), str(state_path))
    assert cfg.base_model == "mlx-community/Qwen2.5-32B-Instruct-4bit"


def test_build_ship_config_applies_best_hyperparams(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = make_state(tmp_path, {"lora_rank": 16, "learning_rate": 2e-4,
                                        "lora_layers": 24, "batch_size": 4})
    cfg = build_ship_config(str(recipe), str(state_path))
    assert cfg.lora_rank == 16
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert cfg.lora_layers == 24


def test_build_ship_config_with_empty_best_config_uses_recipe_defaults(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = make_state(tmp_path, {})
    cfg = build_ship_config(str(recipe), str(state_path))
    assert cfg.base_model == "mlx-community/Qwen2.5-32B-Instruct-4bit"
    assert cfg.lora_rank == 8  # recipe default


# --- apply_to_ship_model ---

def test_apply_to_ship_model_calls_training(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = make_state(tmp_path, {"lora_rank": 16, "learning_rate": 2e-4,
                                        "lora_layers": 24, "batch_size": 4})
    with patch("core.transfer.run_training") as mock_train:
        apply_to_ship_model(str(recipe), str(state_path))
        mock_train.assert_called_once()
        called_cfg = mock_train.call_args[0][0]
        assert called_cfg.base_model == "mlx-community/Qwen2.5-32B-Instruct-4bit"
        assert called_cfg.lora_rank == 16


def test_apply_to_ship_model_raises_on_direct_mode(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base_model: mlx-community/Qwen2.5-7B-Instruct-4bit\n"
        "mode: direct\n"
        "lora_rank: 8\n"
        "lora_alpha: 16.0\n"
        "lora_layers: 16\n"
        "iters: 500\n"
        "learning_rate: 0.0001\n"
        "batch_size: 4\n"
        "grad_checkpoint: true\n"
        "data_dir: data\n"
        "adapter_path: adapters/test\n"
        "fused_path: fused/test\n"
    )
    state_path = make_state(tmp_path, {})
    with pytest.raises(ValueError, match="transfer mode"):
        apply_to_ship_model(str(recipe), str(state_path))
