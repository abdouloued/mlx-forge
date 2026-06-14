# Preparing your data for mlx-forge

mlx-forge trains on **chat JSONL** — one JSON object per line, each containing a `messages` list. This guide explains the required format, shows how to convert data you already have, and covers format rules for each recipe type.

---

## The required format

Every line in `train.jsonl` and `valid.jsonl` must be a JSON object with a `messages` array:

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Translate to French: Hello"}, {"role": "assistant", "content": "Bonjour"}]}
```

### Rules

| Rule | Why |
|---|---|
| `messages` must be a list | Required field — training fails without it |
| At least one `user` turn | The model needs something to respond to |
| Last turn must be `assistant` | That is the text the model is trained to produce |
| Valid roles: `system`, `user`, `assistant`, `tool` | Other roles are rejected by mlx-lm |
| `content` or `tool_calls` required on each message | No empty messages |

### With a system prompt (recommended)

```jsonl
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}
```

### Multi-turn conversation

```jsonl
{"messages": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there! How can I help?"},
  {"role": "user", "content": "What is MLX?"},
  {"role": "assistant", "content": "MLX is Apple's machine-learning framework for Apple Silicon."}
]}
```

---

## Validate your data

Before training, always validate:

```bash
uv run python -m core.datakit validate recipes/toolcalling/data/train.jsonl
```

Output when clean:

```
Total lines: 20
  Valid:  20
  Errors: 0
```

Output with errors:

```
Total lines: 5
  Valid:  3
  Errors: 2
  Line 2: missing required field 'messages'
  Line 4: last message role is 'user' — must be 'assistant'
```

Fix every error before training. A single malformed line can break a training run.

---

## Convert data you already have

### From Q&A pairs (JSONL)

If you have a JSONL file with `input`/`output` fields:

```jsonl
{"input": "What is the capital of France?", "output": "Paris."}
{"input": "What is 2+2?", "output": "4."}
```

Convert it:

```bash
uv run python -m core.datakit convert \
  --from qa \
  --input my_data.jsonl \
  --output recipes/my_recipe/data/train.jsonl \
  --system "You are a helpful assistant."
```

Custom field names (e.g. `question`/`answer`):

```bash
uv run python -m core.datakit convert \
  --from qa \
  --input my_data.jsonl \
  --input-col question \
  --output-col answer \
  --output train.jsonl
```

### From CSV

If you have a spreadsheet or CSV export:

```csv
question,answer
What is the capital of France?,Paris.
What is 2+2?,4.
```

Convert it:

```bash
uv run python -m core.datakit convert \
  --from csv \
  --input my_data.csv \
  --input-col question \
  --output-col answer \
  --output recipes/my_recipe/data/train.jsonl \
  --system "You are a helpful assistant."
```

### From Alpaca / instruction format

The [Alpaca format](https://github.com/tatsu-lab/stanford_alpaca) has `instruction`, optional `input`, and `output`:

```jsonl
{"instruction": "Summarise the following text.", "input": "The sky is blue and vast.", "output": "Blue sky."}
{"instruction": "Write a haiku about autumn.", "input": "", "output": "Leaves fall silently / gold carpet on the forest floor / winter waits ahead"}
```

Convert it:

```bash
uv run python -m core.datakit convert \
  --from instruction \
  --input alpaca_data.jsonl \
  --output train.jsonl \
  --system "You are a helpful assistant."
```

The converter concatenates `instruction` + `input` into the user turn (separated by a blank line when both are present).

---

## Recipe-specific formats

### toolcalling

The tool-calling recipe uses an extended format with a `tools` schema and `tool_calls` in the assistant turn:

```jsonl
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"}
      }]
    }
  ],
  "expected": {
    "name": "get_weather",
    "arguments": {"location": "Paris", "unit": "celsius"}
  }
}
```

The `expected` field is used by the scorer (`eval.py`) and is not seen by the model during training. You need:

- At least 20 training examples
- At least 8 validation examples covering distinct tool calls and argument types
- One tool schema per example (or a shared schema for all examples)

### edge_android

Plain chat format with keyword-tagged examples:

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a compact assistant for Android devices."},
    {"role": "user", "content": "How do I clear app cache on Android 14?"},
    {"role": "assistant", "content": "Go to Settings → Apps → [App name] → Storage → Clear Cache."}
  ],
  "keywords": ["clear cache"],
  "max_words": 50
}
```

The `keywords` field is used by the scorer to check answer correctness; `max_words` sets the conciseness limit. Short, correct answers score 1.0.

### healthcare_coding

Each example includes an expected ICD-10 code (or marks the question as out-of-scope):

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a clinical coding assistant. All data is synthetic."},
    {"role": "user", "content": "A patient presents with acute appendicitis without mention of peritonitis."},
    {"role": "assistant", "content": "K35.80"}
  ],
  "expected_code": "K35.80",
  "out_of_scope": false
}
```

For questions the model should decline to answer:

```jsonl
{
  "messages": [
    {"role": "user", "content": "Prescribe antibiotics for this patient."},
    {"role": "assistant", "content": "I cannot provide prescriptions. Please consult a licensed physician."}
  ],
  "expected_code": null,
  "out_of_scope": true
}
```

**Important:** All healthcare training data must be synthetic. If you substitute real clinical data, ensure it is de-identified and verify compliance with applicable regulations (HIPAA, etc.).

---

## Data size guidelines

| Recipe | Minimum train | Recommended train | Minimum valid |
|---|---|---|---|
| toolcalling | 20 | 100–500 | 8 |
| edge_android | 50 | 200–1000 | 20 |
| healthcare_coding | 50 | 200–500 | 20 |
| custom recipe | 20 | 100+ | 8 |

More data does not always help — diverse data matters more than volume. A well-curated set of 50 training examples often outperforms 500 noisy ones.

---

## Data quality checklist

Before training, verify:

- [ ] `uv run python -m core.datakit validate` passes with 0 errors
- [ ] Every training example demonstrates the exact behaviour you want (garbage in = garbage out)
- [ ] Validation examples come from a different source than training examples (no overlap)
- [ ] The validation set covers the full range of inputs the model will see in production
- [ ] For tool-calling: `expected.arguments` values match what the `arguments` JSON string contains
- [ ] For healthcare: all data is synthetic or properly de-identified

---

## Splitting into train / valid

If you have one big file and need to split it:

```python
import json, random, pathlib

data = [json.loads(l) for l in pathlib.Path("all_data.jsonl").read_text().splitlines() if l.strip()]
random.shuffle(data)
split = int(len(data) * 0.9)

pathlib.Path("data/train.jsonl").write_text("\n".join(json.dumps(ex) for ex in data[:split]) + "\n")
pathlib.Path("data/valid.jsonl").write_text("\n".join(json.dumps(ex) for ex in data[split:]) + "\n")
```

Aim for a 90/10 train/valid split. If you have fewer than 40 examples, keep at least 8 for validation regardless of the ratio.
