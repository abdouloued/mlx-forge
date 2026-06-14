from unittest.mock import patch, MagicMock
import pytest
from core.config import RecipeConfig
from core.fuse import build_fuse_command, fuse_adapter


def make_cfg(**kwargs) -> RecipeConfig:
    defaults = dict(
        base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="adapters/run1",
        fused_path="fused/run1",
    )
    defaults.update(kwargs)
    return RecipeConfig(**defaults)


def test_fuse_command_includes_model():
    cmd = build_fuse_command(make_cfg())
    assert "mlx-community/Qwen2.5-7B-Instruct-4bit" in " ".join(cmd)


def test_fuse_command_uses_new_entrypoint():
    cmd = build_fuse_command(make_cfg())
    joined = " ".join(cmd)
    assert "mlx_lm.fuse" not in joined
    assert "mlx_lm" in joined and "fuse" in joined


def test_fuse_command_includes_adapter_path():
    cmd = build_fuse_command(make_cfg(adapter_path="adapters/run1"))
    assert "adapters/run1" in " ".join(cmd)


def test_fuse_command_includes_save_path():
    cmd = build_fuse_command(make_cfg(fused_path="fused/run1"))
    assert "fused/run1" in " ".join(cmd)


def test_fuse_adapter_calls_subprocess():
    cfg = make_cfg()
    with patch("core.fuse.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        fuse_adapter(cfg)
        mock_run.assert_called_once()


def test_fuse_adapter_raises_on_failure():
    cfg = make_cfg()
    with patch("core.fuse.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        with pytest.raises(RuntimeError, match="Fusion failed"):
            fuse_adapter(cfg)
