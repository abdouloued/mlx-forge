import json
import random
import dataclasses
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.config import RecipeConfig
from core.loop import (
    SEARCH_SPACE,
    LoopState,
    propose_config,
    save_state,
    load_state,
    update_program_md,
)


def base_cfg() -> RecipeConfig:
    return RecipeConfig(base_model="mlx-community/Qwen2.5-7B-Instruct-4bit")


# --- propose_config ---

def test_propose_config_changes_exactly_one_param():
    cfg = base_cfg()
    new_cfg = propose_config(cfg, SEARCH_SPACE, random.Random(42))
    changed = sum([
        cfg.learning_rate != new_cfg.learning_rate,
        cfg.lora_rank != new_cfg.lora_rank,
        cfg.lora_layers != new_cfg.lora_layers,
        cfg.batch_size != new_cfg.batch_size,
    ])
    assert changed == 1


def test_propose_config_preserves_base_model():
    cfg = base_cfg()
    new_cfg = propose_config(cfg, SEARCH_SPACE, random.Random(0))
    assert new_cfg.base_model == cfg.base_model


def test_propose_config_value_in_search_space():
    cfg = base_cfg()
    for seed in range(20):
        new_cfg = propose_config(cfg, SEARCH_SPACE, random.Random(seed))
        assert new_cfg.learning_rate in SEARCH_SPACE["learning_rate"]
        assert new_cfg.lora_rank in SEARCH_SPACE["lora_rank"]
        assert new_cfg.lora_layers in SEARCH_SPACE["lora_layers"]
        assert new_cfg.batch_size in SEARCH_SPACE["batch_size"]


def test_propose_config_does_not_mutate_original():
    cfg = base_cfg()
    original_lr = cfg.learning_rate
    propose_config(cfg, SEARCH_SPACE, random.Random(0))
    assert cfg.learning_rate == original_lr


# --- LoopState serialization ---

def test_state_save_and_load(tmp_path):
    state = LoopState(best_score=0.75, experiment=3)
    path = tmp_path / "loop_state.json"
    save_state(state, path)
    loaded = load_state(path)
    assert loaded.best_score == 0.75
    assert loaded.experiment == 3


def test_load_state_missing_file_returns_default(tmp_path):
    path = tmp_path / "nonexistent.json"
    state = load_state(path)
    assert state.best_score == 0.0
    assert state.experiment == 0


def test_state_round_trips_best_config(tmp_path):
    cfg_dict = {"lora_rank": 16, "learning_rate": 2e-4}
    state = LoopState(best_score=0.8, experiment=5, best_config=cfg_dict)
    path = tmp_path / "state.json"
    save_state(state, path)
    loaded = load_state(path)
    assert loaded.best_config == cfg_dict


# --- update_program_md ---

def test_update_program_md_updates_score(tmp_path):
    md_path = tmp_path / "program.md"
    md_path.write_text("## Current best score\nscore: 0.0\nexperiment: baseline\n")
    update_program_md(md_path, score=0.875, experiment=4)
    content = md_path.read_text()
    assert "0.875" in content
    assert "experiment: 4" in content


def test_update_program_md_leaves_other_content_intact(tmp_path):
    md_path = tmp_path / "program.md"
    md_path.write_text(
        "# Title\n\n## Current best score\nscore: 0.0\nexperiment: baseline\n\n## Notes\nkeep this\n"
    )
    update_program_md(md_path, score=0.5, experiment=2)
    content = md_path.read_text()
    assert "keep this" in content
    assert "# Title" in content
