"""Tests for the data-flywheel. No model required."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from recipes.data_flywheel.flywheel import (
    judge_candidates,
    append_to_training_data,
    format_as_training_example,
    FlywheelRound,
)


# --- judge_candidates ---

def make_candidate(response: str, seed: dict, score: float) -> dict:
    return {"response": response, "seed": seed, "score": score}


def test_judge_keeps_above_threshold():
    candidates = [
        make_candidate("good", {}, 0.90),
        make_candidate("bad", {}, 0.40),
        make_candidate("ok", {}, 0.80),
    ]
    kept = judge_candidates(candidates, threshold=0.75)
    assert len(kept) == 2
    assert all(c["score"] >= 0.75 for c in kept)


def test_judge_rejects_all_below_threshold():
    candidates = [make_candidate("bad", {}, 0.20), make_candidate("worse", {}, 0.10)]
    assert judge_candidates(candidates, threshold=0.75) == []


def test_judge_keeps_all_above_threshold():
    candidates = [make_candidate("a", {}, 1.0), make_candidate("b", {}, 0.9)]
    assert len(judge_candidates(candidates, threshold=0.80)) == 2


def test_judge_exact_threshold_is_kept():
    candidates = [make_candidate("x", {}, 0.75)]
    assert len(judge_candidates(candidates, threshold=0.75)) == 1


# --- format_as_training_example ---

def test_format_preserves_messages():
    seed = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "tools": [],
    }
    example = format_as_training_example(seed, response='{"name":"calc","arguments":{"expression":"2+2"}}')
    assert example["messages"][-1]["role"] == "assistant"
    assert '{"name":"calc"' in example["messages"][-1]["content"]


def test_format_appends_assistant_turn():
    seed = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
    example = format_as_training_example(seed, response="hello")
    messages = example["messages"]
    assert messages[-1] == {"role": "assistant", "content": "hello"}


def test_format_includes_tools():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    seed = {"messages": [{"role": "user", "content": "weather?"}], "tools": tools}
    example = format_as_training_example(seed, response="sunny")
    assert example["tools"] == tools


# --- append_to_training_data ---

def test_append_writes_valid_jsonl(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("")
    examples = [
        {"messages": [{"role": "user", "content": "q1"}], "tools": []},
        {"messages": [{"role": "user", "content": "q2"}], "tools": []},
    ]
    count = append_to_training_data(examples, train_file)
    assert count == 2
    lines = [json.loads(l) for l in train_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_append_does_not_overwrite_existing(tmp_path):
    train_file = tmp_path / "train.jsonl"
    existing = {"messages": [{"role": "user", "content": "original"}], "tools": []}
    train_file.write_text(json.dumps(existing) + "\n")
    new_examples = [{"messages": [{"role": "user", "content": "new"}], "tools": []}]
    append_to_training_data(new_examples, train_file)
    lines = [json.loads(l) for l in train_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["messages"][0]["content"] == "original"


def test_append_returns_zero_for_empty_list(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("")
    assert append_to_training_data([], train_file) == 0


# --- FlywheelRound dataclass ---

def test_flywheel_round_tracks_stats():
    r = FlywheelRound(round_n=1, generated=10, kept=3, train_size_before=20, train_size_after=23)
    assert r.kept_rate == pytest.approx(0.30)
