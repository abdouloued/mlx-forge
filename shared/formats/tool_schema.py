"""
Tool call schema validation for the toolcalling recipe.

Expected format (assistant turn content, JSON-encoded):
  {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}

The scorer in recipes/toolcalling/scorer.py calls validate_tool_call() before
checking argument values, so any format error scores 0.0 cleanly.
"""
from __future__ import annotations

from typing import Any


class ToolCallError(ValueError):
    pass


def validate_tool_call(call: Any) -> None:
    """Validate that `call` has the structure {name: str, arguments: dict}."""
    if not isinstance(call, dict):
        raise ToolCallError(f"tool call must be a dict, got {type(call).__name__}")
    if "name" not in call:
        raise ToolCallError("tool call missing 'name'")
    if not isinstance(call["name"], str):
        raise ToolCallError("tool call 'name' must be a string")
    if "arguments" not in call:
        raise ToolCallError("tool call missing 'arguments'")
    if not isinstance(call["arguments"], dict):
        raise ToolCallError("tool call 'arguments' must be a dict")
