"""Tests for core/datakit — data validation and format conversion."""
import json
import pytest
from pathlib import Path
from core.datakit import (
    validate_example,
    validate_jsonl,
    convert_qa_pairs,
    convert_csv,
    convert_instruction_pairs,
    ValidationReport,
    ConversionResult,
)


# ── validate_example ──────────────────────────────────────────────────────────

def test_valid_chat_example():
    ex = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    errors = validate_example(ex)
    assert errors == []


def test_missing_messages_field():
    errors = validate_example({"text": "hi"})
    assert any("messages" in e for e in errors)


def test_messages_not_a_list():
    errors = validate_example({"messages": "not a list"})
    assert any("list" in e for e in errors)


def test_empty_messages():
    errors = validate_example({"messages": []})
    assert any("empty" in e.lower() for e in errors)


def test_message_missing_role():
    errors = validate_example({"messages": [{"content": "hi"}]})
    assert any("role" in e for e in errors)


def test_message_missing_content():
    errors = validate_example({"messages": [{"role": "user"}]})
    assert any("content" in e for e in errors)


def test_invalid_role():
    errors = validate_example({"messages": [{"role": "boss", "content": "hi"}]})
    assert any("role" in e for e in errors)


def test_no_user_turn():
    errors = validate_example({"messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Hello!"},
    ]})
    assert any("user" in e for e in errors)


def test_last_turn_not_assistant():
    errors = validate_example({"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "Are you there?"},
    ]})
    assert any("assistant" in e for e in errors)


def test_tool_calls_without_tools_field_warns():
    ex = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]},
        ]
    }
    errors = validate_example(ex)
    assert any("tools" in e for e in errors)


def test_valid_tool_call_example():
    ex = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}]
    }
    assert validate_example(ex) == []


# ── validate_jsonl ─────────────────────────────────────────────────────────────

def test_validate_jsonl_all_valid(tmp_path):
    f = tmp_path / "train.jsonl"
    good = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}
    f.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    report = validate_jsonl(f)
    assert report.valid_count == 2
    assert report.error_count == 0
    assert report.is_clean


def test_validate_jsonl_counts_errors(tmp_path):
    f = tmp_path / "train.jsonl"
    good = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}
    bad = {"text": "wrong format"}
    f.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    report = validate_jsonl(f)
    assert report.valid_count == 1
    assert report.error_count == 1
    assert not report.is_clean


def test_validate_jsonl_catches_invalid_json(tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text("{valid: false, broken\n")
    report = validate_jsonl(f)
    assert report.error_count == 1


def test_validate_jsonl_skips_blank_lines(tmp_path):
    f = tmp_path / "train.jsonl"
    good = json.dumps({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]})
    f.write_text(good + "\n\n" + good + "\n")
    report = validate_jsonl(f)
    assert report.valid_count == 2


# ── convert_qa_pairs ──────────────────────────────────────────────────────────

def test_convert_qa_pairs_basic():
    pairs = [
        {"input": "What is 2+2?", "output": "4."},
        {"input": "Capital of France?", "output": "Paris."},
    ]
    result = convert_qa_pairs(pairs, system_prompt="You are helpful.")
    assert result.count == 2
    assert result.errors == []
    ex = result.examples[0]
    assert ex["messages"][0]["role"] == "system"
    assert ex["messages"][1]["role"] == "user"
    assert ex["messages"][2]["role"] == "assistant"
    assert ex["messages"][2]["content"] == "4."


def test_convert_qa_pairs_without_system():
    pairs = [{"input": "hi", "output": "hello"}]
    result = convert_qa_pairs(pairs)
    assert result.examples[0]["messages"][0]["role"] == "user"


def test_convert_qa_pairs_skips_empty_input():
    pairs = [
        {"input": "", "output": "something"},
        {"input": "valid", "output": "answer"},
    ]
    result = convert_qa_pairs(pairs)
    assert result.count == 1
    assert result.skipped == 1


def test_convert_qa_pairs_skips_empty_output():
    pairs = [{"input": "question", "output": ""}]
    result = convert_qa_pairs(pairs)
    assert result.count == 0
    assert result.skipped == 1


def test_convert_qa_pairs_custom_keys():
    pairs = [{"question": "hi", "answer": "hello"}]
    result = convert_qa_pairs(pairs, input_key="question", output_key="answer")
    assert result.count == 1


# ── convert_csv ───────────────────────────────────────────────────────────────

def test_convert_csv_basic(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("input,output\nWhat is 2+2?,4\nCapital of France?,Paris\n")
    result = convert_csv(csv_file, input_col="input", output_col="output",
                         system_prompt="You are helpful.")
    assert result.count == 2
    assert result.examples[0]["messages"][-1]["content"] == "4"


def test_convert_csv_missing_column(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("question,answer\nhi,hello\n")
    result = convert_csv(csv_file, input_col="input", output_col="output")
    assert result.count == 0
    assert len(result.errors) > 0


# ── convert_instruction_pairs ────────────────────────────────────────────────

def test_convert_instruction_pairs():
    pairs = [
        {"instruction": "Summarise this text.", "input": "The sky is blue.", "output": "Blue sky."},
        {"instruction": "Translate to French.", "input": "Hello", "output": "Bonjour"},
    ]
    result = convert_instruction_pairs(pairs)
    assert result.count == 2
    user_msg = result.examples[0]["messages"][0]["content"]
    assert "Summarise" in user_msg
    assert "blue" in user_msg


def test_convert_instruction_without_input():
    pairs = [{"instruction": "Write a haiku.", "output": "Old pond / frog jumps in / water's sound"}]
    result = convert_instruction_pairs(pairs)
    assert result.count == 1
    assert "haiku" in result.examples[0]["messages"][0]["content"]


# ── ValidationReport ──────────────────────────────────────────────────────────

def test_validation_report_summary():
    report = ValidationReport(
        total=10, valid_count=8, error_count=2,
        line_errors={3: ["missing messages"], 7: ["no user turn"]}
    )
    summary = report.summary()
    assert "8" in summary
    assert "2" in summary
