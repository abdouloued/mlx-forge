"""Data validator and format converter for mlx-forge training data."""
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

VALID_ROLES = {"system", "user", "assistant", "tool"}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    total: int
    valid_count: int
    error_count: int
    line_errors: dict[int, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0

    def summary(self) -> str:
        lines = [
            f"Total lines: {self.total}",
            f"  Valid:  {self.valid_count}",
            f"  Errors: {self.error_count}",
        ]
        for lineno, errs in sorted(self.line_errors.items()):
            for e in errs:
                lines.append(f"  Line {lineno}: {e}")
        return "\n".join(lines)


@dataclass
class ConversionResult:
    examples: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def count(self) -> int:
        return len(self.examples)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_example(ex: dict) -> list[str]:
    """Return a list of error strings for one training example (empty = valid)."""
    errors: list[str] = []

    if "messages" not in ex:
        errors.append("missing required field 'messages'")
        return errors

    msgs = ex["messages"]
    if not isinstance(msgs, list):
        errors.append("'messages' must be a list")
        return errors

    if len(msgs) == 0:
        errors.append("'messages' is empty — must have at least one turn")
        return errors

    has_user = False
    has_tool_calls = False
    for i, msg in enumerate(msgs):
        pos = f"messages[{i}]"
        if not isinstance(msg, dict):
            errors.append(f"{pos}: must be a dict, got {type(msg).__name__}")
            continue
        if "role" not in msg:
            errors.append(f"{pos}: missing required key 'role'")
        elif msg["role"] not in VALID_ROLES:
            errors.append(
                f"{pos}: invalid role {msg['role']!r} — "
                f"must be one of {sorted(VALID_ROLES)}"
            )
        if msg.get("role") == "user":
            has_user = True
        if "content" not in msg and "tool_calls" not in msg:
            errors.append(f"{pos}: must have 'content' or 'tool_calls'")
        if "tool_calls" in msg:
            has_tool_calls = True

    if not has_user:
        errors.append("no 'user' turn found — every example needs at least one user message")

    last_role = msgs[-1].get("role") if msgs else None
    if last_role != "assistant":
        errors.append(
            f"last message role is {last_role!r} — must be 'assistant' "
            "(the target the model is trained to produce)"
        )

    if has_tool_calls and "tools" not in ex:
        errors.append(
            "assistant uses 'tool_calls' but the example has no top-level 'tools' list — "
            "mlx-lm requires the tool schema in each example"
        )

    return errors


def validate_jsonl(path: str | Path) -> ValidationReport:
    """Validate every line in a JSONL file and return a ValidationReport."""
    path = Path(path)
    total = valid = errors = 0
    line_errors: dict[int, list[str]] = {}

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        total += 1
        try:
            ex = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors += 1
            line_errors[lineno] = [f"invalid JSON — {exc}"]
            continue
        errs = validate_example(ex)
        if errs:
            errors += 1
            line_errors[lineno] = errs
        else:
            valid += 1

    return ValidationReport(
        total=total,
        valid_count=valid,
        error_count=errors,
        line_errors=line_errors,
    )


# ── Converters ─────────────────────────────────────────────────────────────────

def _make_messages(
    user_text: str,
    assistant_text: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    msgs.append({"role": "assistant", "content": assistant_text})
    return msgs


def convert_qa_pairs(
    pairs: list[dict],
    system_prompt: Optional[str] = None,
    input_key: str = "input",
    output_key: str = "output",
) -> ConversionResult:
    """Convert Q&A dicts ({input, output}) to chat JSONL format."""
    result = ConversionResult()
    for i, pair in enumerate(pairs):
        user_text = str(pair.get(input_key, "")).strip()
        assistant_text = str(pair.get(output_key, "")).strip()
        if not user_text or not assistant_text:
            result.skipped += 1
            continue
        result.examples.append(
            {"messages": _make_messages(user_text, assistant_text, system_prompt)}
        )
    return result


def convert_csv(
    path: str | Path,
    input_col: str,
    output_col: str,
    system_prompt: Optional[str] = None,
) -> ConversionResult:
    """Convert a CSV file to chat JSONL using the specified column names."""
    path = Path(path)
    result = ConversionResult()
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                if input_col not in row or output_col not in row:
                    missing = [c for c in (input_col, output_col) if c not in row]
                    result.errors.append(
                        f"row {i}: column(s) not found: {missing} "
                        f"— available: {list(row.keys())}"
                    )
                    continue
                user_text = str(row[input_col]).strip()
                assistant_text = str(row[output_col]).strip()
                if not user_text or not assistant_text:
                    result.skipped += 1
                    continue
                result.examples.append(
                    {"messages": _make_messages(user_text, assistant_text, system_prompt)}
                )
    except FileNotFoundError:
        result.errors.append(f"file not found: {path}")
    except Exception as exc:
        result.errors.append(f"could not read CSV: {exc}")
    return result


def convert_instruction_pairs(
    pairs: list[dict],
    system_prompt: Optional[str] = None,
) -> ConversionResult:
    """Convert Alpaca-style dicts ({instruction, input?, output}) to chat JSONL."""
    result = ConversionResult()
    for i, pair in enumerate(pairs):
        instruction = str(pair.get("instruction", "")).strip()
        context = str(pair.get("input", "")).strip()
        assistant_text = str(pair.get("output", "")).strip()
        if not instruction or not assistant_text:
            result.skipped += 1
            continue
        user_text = f"{instruction}\n\n{context}" if context else instruction
        result.examples.append(
            {"messages": _make_messages(user_text, assistant_text, system_prompt)}
        )
    return result


def save_jsonl(examples: list[dict], path: str | Path) -> int:
    """Write examples to a JSONL file, return count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli_validate(args: list[str]) -> None:
    if not args:
        print("usage: python -m core.datakit validate <file.jsonl>", file=sys.stderr)
        sys.exit(1)
    report = validate_jsonl(args[0])
    print(report.summary())
    if not report.is_clean:
        sys.exit(1)


def _cli_convert(args: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m core.datakit convert",
        description="Convert raw data to mlx-forge chat JSONL format.",
    )
    p.add_argument("--from", dest="fmt", required=True,
                   choices=["qa", "csv", "instruction"],
                   help="input format")
    p.add_argument("--input", required=True, help="input file (.jsonl or .csv)")
    p.add_argument("--output", required=True, help="output .jsonl path")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--input-col", default="input",
                   help="CSV/JSONL column for the user turn (default: input)")
    p.add_argument("--output-col", default="output",
                   help="CSV/JSONL column for the assistant turn (default: output)")
    parsed = p.parse_args(args)

    if parsed.fmt == "csv":
        result = convert_csv(
            parsed.input, parsed.input_col, parsed.output_col, parsed.system
        )
    elif parsed.fmt == "qa":
        raw = [json.loads(l) for l in Path(parsed.input).read_text().splitlines() if l.strip()]
        result = convert_qa_pairs(raw, parsed.system, parsed.input_col, parsed.output_col)
    else:
        raw = [json.loads(l) for l in Path(parsed.input).read_text().splitlines() if l.strip()]
        result = convert_instruction_pairs(raw, parsed.system)

    if result.errors:
        for e in result.errors:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    count = save_jsonl(result.examples, parsed.output)
    skipped = result.skipped
    print(f"wrote {count} examples to {parsed.output}" +
          (f" ({skipped} skipped)" if skipped else ""))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("validate", "convert"):
        print(
            "mlx-forge datakit — data validator & format converter\n\n"
            "Commands:\n"
            "  validate <file.jsonl>          check every line for format errors\n"
            "  convert --from qa|csv|instruction\n"
            "          --input <file>         source file\n"
            "          --output <file.jsonl>  destination\n"
            "          [--system 'You are …'] optional system prompt\n"
            "          [--input-col <col>]    column name for user turn  (default: input)\n"
            "          [--output-col <col>]   column name for model turn (default: output)\n\n"
            "Examples:\n"
            "  uv run python -m core.datakit validate recipes/toolcalling/data/train.jsonl\n"
            "  uv run python -m core.datakit convert --from csv --input my_data.csv \\\n"
            "    --input-col question --output-col answer --output train.jsonl\n"
            "  uv run python -m core.datakit convert --from instruction --input alpaca.jsonl \\\n"
            "    --system 'You are a helpful assistant.' --output train.jsonl",
            file=sys.stderr,
        )
        sys.exit(0)

    cmd, *rest = sys.argv[1:]
    if cmd == "validate":
        _cli_validate(rest)
    else:
        _cli_convert(rest)


if __name__ == "__main__":
    main()
