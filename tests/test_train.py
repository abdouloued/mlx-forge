import subprocess
from unittest.mock import patch, MagicMock
import pytest
from core.config import RecipeConfig
from core.train import build_train_command, run_training


def make_cfg(**kwargs) -> RecipeConfig:
    defaults = dict(
        base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        lora_rank=8,
        lora_layers=16,
        iters=100,
        learning_rate=1e-4,
        batch_size=4,
        grad_checkpoint=True,
        data_dir="recipes/toolcalling/data",
        adapter_path="adapters/test",
    )
    defaults.update(kwargs)
    return RecipeConfig(**defaults)


def test_command_includes_model():
    cmd = build_train_command(make_cfg())
    assert "mlx-community/Qwen2.5-7B-Instruct-4bit" in cmd


def test_command_includes_iters():
    cmd = build_train_command(make_cfg(iters=500))
    assert "500" in cmd


def test_command_includes_num_layers():
    cmd = build_train_command(make_cfg(lora_layers=16))
    assert "--num-layers" in cmd
    assert "16" in cmd


def test_rank_goes_to_config_file():
    # rank is now written to a temp YAML via -c, not passed as a CLI flag
    cmd = build_train_command(make_cfg(lora_rank=16))
    assert "--rank" not in cmd
    assert "-c" in cmd


def test_command_includes_data_dir():
    cmd = build_train_command(make_cfg(data_dir="recipes/toolcalling/data"))
    assert "recipes/toolcalling/data" in cmd


def test_command_includes_adapter_path():
    cmd = build_train_command(make_cfg(adapter_path="adapters/run1"))
    assert "adapters/run1" in cmd


def test_command_uses_new_entrypoint():
    cmd = build_train_command(make_cfg())
    # new form: python -m mlx_lm lora  (not python -m mlx_lm.lora)
    joined = " ".join(cmd)
    assert "mlx_lm.lora" not in joined
    assert "mlx_lm" in joined and "lora" in joined


def test_run_training_calls_subprocess():
    cfg = make_cfg()
    with patch("core.train.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_training(cfg)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert isinstance(cmd, list)


def test_run_training_raises_on_failure():
    cfg = make_cfg()
    with patch("core.train.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        with pytest.raises(RuntimeError, match="Training failed"):
            run_training(cfg)
