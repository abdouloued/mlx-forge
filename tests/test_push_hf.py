import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from core.push_hf import push_to_hf, build_repo_card


def test_build_repo_card_contains_repo_id():
    card = build_repo_card(
        repo_id="myuser/my-model",
        base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        recipe="toolcalling",
    )
    assert "myuser/my-model" in card


def test_build_repo_card_contains_base_model():
    card = build_repo_card(
        repo_id="myuser/my-model",
        base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        recipe="toolcalling",
    )
    assert "mlx-community/Qwen2.5-7B-Instruct-4bit" in card


def test_build_repo_card_contains_recipe():
    card = build_repo_card(
        repo_id="myuser/my-model",
        base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        recipe="toolcalling",
    )
    assert "toolcalling" in card


def test_push_to_hf_calls_upload(tmp_path):
    model_dir = tmp_path / "fused"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with patch("core.push_hf.upload_folder") as mock_upload, \
         patch("core.push_hf.create_repo"):
        mock_upload.return_value = MagicMock()
        push_to_hf(
            fused_path=str(model_dir),
            repo_id="user/model",
            base_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            recipe="toolcalling",
            private=True,
        )
        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs["repo_id"] == "user/model"


def test_push_to_hf_creates_repo_with_private_flag(tmp_path):
    model_dir = tmp_path / "fused"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with patch("core.push_hf.upload_folder") as mock_upload, \
         patch("core.push_hf.create_repo") as mock_create:
        mock_upload.return_value = MagicMock()
        push_to_hf(
            fused_path=str(model_dir),
            repo_id="user/model",
            base_model="any/model",
            recipe="toolcalling",
            private=True,
        )
        mock_create.assert_called_once()
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs.get("private") is True
