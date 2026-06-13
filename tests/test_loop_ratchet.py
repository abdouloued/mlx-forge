"""Integration-level tests for the ratchet_loop function using mocked train+score."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.loop import ratchet_loop, load_state


def make_recipe_yaml(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base_model: mlx-community/Qwen2.5-7B-Instruct-4bit\n"
        "mode: direct\n"
        "lora_rank: 8\n"
        "lora_alpha: 16.0\n"
        "lora_layers: 16\n"
        "iters: 100\n"
        "learning_rate: 0.0001\n"
        "batch_size: 4\n"
        "grad_checkpoint: true\n"
        "data_dir: data\n"
        "adapter_path: adapters/test\n"
        "fused_path: fused/test\n"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "valid.jsonl").write_text(
        '{"messages":[],"expected":{"name":"get_weather","arguments":{"location":"Paris"}}}\n'
    )
    program_md = tmp_path / "program.md"
    program_md.write_text("## Current best score\nscore: 0.0\nexperiment: baseline\n")
    return recipe


def test_ratchet_keeps_better_experiment(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = tmp_path / "state.json"

    scores = [0.75, 0.50]  # first experiment better, second worse
    score_iter = iter(scores)

    with patch("core.loop.run_training"), \
         patch("core.loop._score_model", side_effect=lambda *_: next(score_iter)), \
         patch("core.loop._git_commit"):
        ratchet_loop(
            recipe_path=str(recipe),
            n_experiments=2,
            target_score=0.99,
            state_path=str(state_path),
            seed=0,
        )

    state = load_state(state_path)
    assert state.best_score == 0.75


def test_ratchet_stops_at_target_score(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = tmp_path / "state.json"

    with patch("core.loop.run_training"), \
         patch("core.loop._score_model", return_value=0.95), \
         patch("core.loop._git_commit"):
        ratchet_loop(
            recipe_path=str(recipe),
            n_experiments=10,
            target_score=0.90,
            state_path=str(state_path),
            seed=0,
        )

    state = load_state(state_path)
    assert state.best_score >= 0.90
    assert state.experiment == 1  # stopped after first experiment hit target


def test_ratchet_discards_worse_experiment(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = tmp_path / "state.json"

    commit_calls = []
    with patch("core.loop.run_training"), \
         patch("core.loop._score_model", return_value=0.30), \
         patch("core.loop._git_commit", side_effect=lambda *a, **kw: commit_calls.append(a)):
        ratchet_loop(
            recipe_path=str(recipe),
            n_experiments=3,
            target_score=0.99,
            state_path=str(state_path),
            seed=0,
        )

    # No experiment beat 0.0 baseline with score 0.30
    # Wait - 0.30 > 0.0 so first will be kept. Let me use 0.0 as score.
    # This test is checking that non-improving experiments don't commit.
    # All 0.30 experiments beat 0.0 initially. Let me rethink.
    # Actually first exp (0.30 > 0.0) → kept, second (0.30 == 0.30) → discarded.
    assert len(commit_calls) == 1  # only first experiment committed


def test_ratchet_handles_training_failure_gracefully(tmp_path):
    recipe = make_recipe_yaml(tmp_path)
    state_path = tmp_path / "state.json"

    with patch("core.loop.run_training", side_effect=RuntimeError("Training failed")), \
         patch("core.loop._git_commit"):
        ratchet_loop(
            recipe_path=str(recipe),
            n_experiments=2,
            target_score=0.99,
            state_path=str(state_path),
            seed=0,
        )

    state = load_state(state_path)
    assert state.best_score == 0.0  # no improvement since training always fails
    assert state.experiment == 2    # but experiment counter advances
