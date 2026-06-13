import pytest
from shared.formats.tool_schema import validate_tool_call, ToolCallError


def test_valid_call():
    call = {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
    validate_tool_call(call)  # no exception


def test_missing_name():
    with pytest.raises(ToolCallError):
        validate_tool_call({"arguments": {"location": "Paris"}})


def test_missing_arguments():
    with pytest.raises(ToolCallError):
        validate_tool_call({"name": "get_weather"})


def test_arguments_not_dict():
    with pytest.raises(ToolCallError):
        validate_tool_call({"name": "get_weather", "arguments": "Paris"})


def test_name_not_string():
    with pytest.raises(ToolCallError):
        validate_tool_call({"name": 42, "arguments": {}})
