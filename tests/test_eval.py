import json
import pytest
from recipes.toolcalling.eval import score_response, parse_tool_call


# --- parse_tool_call tests ---

def test_parse_valid_json_tool_call():
    response = json.dumps({
        "name": "get_weather",
        "arguments": {"location": "Paris", "unit": "celsius"}
    })
    result = parse_tool_call(response)
    assert result is not None
    assert result["name"] == "get_weather"
    assert result["arguments"]["location"] == "Paris"


def test_parse_plain_text_returns_none():
    result = parse_tool_call("The weather in Paris is sunny.")
    assert result is None


def test_parse_invalid_json_returns_none():
    result = parse_tool_call("{name: get_weather}")
    assert result is None


def test_parse_json_without_name_returns_none():
    result = parse_tool_call('{"arguments": {"location": "Paris"}}')
    assert result is None


# --- score_response tests ---

def make_example(expected_name: str, expected_args: dict) -> dict:
    return {"expected": {"name": expected_name, "arguments": expected_args}}


def test_perfect_score():
    response = json.dumps({
        "name": "get_weather",
        "arguments": {"location": "Paris", "unit": "celsius"}
    })
    example = make_example("get_weather", {"location": "Paris"})
    assert score_response(response, example) == 1.0


def test_correct_function_wrong_required_arg():
    response = json.dumps({
        "name": "get_weather",
        "arguments": {"location": "London"}
    })
    example = make_example("get_weather", {"location": "Paris"})
    score = score_response(response, example)
    assert 0.0 < score < 1.0


def test_wrong_function_name():
    response = json.dumps({
        "name": "web_search",
        "arguments": {"query": "weather Paris"}
    })
    example = make_example("get_weather", {"location": "Paris"})
    score = score_response(response, example)
    assert score < 0.6


def test_plain_text_zero_score():
    response = "The weather in Paris is sunny and warm."
    example = make_example("get_weather", {"location": "Paris"})
    assert score_response(response, example) == 0.0


def test_missing_required_arg():
    response = json.dumps({
        "name": "get_weather",
        "arguments": {}
    })
    example = make_example("get_weather", {"location": "Paris"})
    score = score_response(response, example)
    assert 0.4 < score < 0.9
