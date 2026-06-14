"""mlx-forge interactive TUI — navigate with arrows, fill fields, press Enter to run."""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Footer, Header, Input, Label,
    ListItem, ListView, ProgressBar, RichLog, Static, Rule,
)

_ITER_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")


# ── Action definitions ─────────────────────────────────────────────────────────

@dataclass
class Field:
    id: str
    label: str
    placeholder: str = ""
    default: str = ""
    password: bool = False


@dataclass
class Action:
    name: str
    label: str
    icon: str
    description: str
    fields: list[Field] = field(default_factory=list)


ACTIONS: list[Action] = [
    Action(
        name="download",
        label="Download Model",
        icon="⬇",
        description="Pull a model from Hugging Face and cache it locally.",
        fields=[
            Field("model_id", "Model ID",
                  placeholder="mlx-community/Qwen2.5-7B-Instruct-4bit",
                  default="mlx-community/Qwen2.5-7B-Instruct-4bit"),
        ],
    ),
    Action(
        name="train",
        label="Train",
        icon="⚡",
        description="Fine-tune with LoRA. Progress streams live below.",
        fields=[
            Field("recipe",  "Recipe path",
                  placeholder="recipes/toolcalling/recipe.yaml",
                  default="recipes/toolcalling/recipe.yaml"),
            Field("iters",   "Iters override", placeholder="leave blank to use recipe value"),
            Field("rank",    "LoRA rank override", placeholder="leave blank to use recipe value"),
        ],
    ),
    Action(
        name="eval",
        label="Evaluate",
        icon="📊",
        description="Score the trained adapter on the validation set.",
        fields=[
            Field("recipe", "Recipe path",
                  placeholder="recipes/toolcalling/recipe.yaml",
                  default="recipes/toolcalling/recipe.yaml"),
        ],
    ),
    Action(
        name="fuse",
        label="Fuse Adapter",
        icon="🔗",
        description="Merge the LoRA adapter into full model weights.",
        fields=[
            Field("recipe", "Recipe path",
                  placeholder="recipes/toolcalling/recipe.yaml",
                  default="recipes/toolcalling/recipe.yaml"),
        ],
    ),
    Action(
        name="loop",
        label="Auto-search Loop",
        icon="🔄",
        description="Overnight ratchet — propose → train → score → keep if better.",
        fields=[
            Field("recipe",       "Recipe path",
                  placeholder="recipes/toolcalling/recipe.yaml",
                  default="recipes/toolcalling/recipe.yaml"),
            Field("n_experiments","Experiments", placeholder="10", default="10"),
            Field("target_score", "Target score", placeholder="0.90", default="0.90"),
        ],
    ),
    Action(
        name="export",
        label="Export GGUF",
        icon="📦",
        description="Convert fused weights to GGUF + generate Ollama Modelfile.",
        fields=[
            Field("recipe",    "Recipe path",
                  placeholder="recipes/toolcalling/recipe.yaml",
                  default="recipes/toolcalling/recipe.yaml"),
            Field("llama_cpp", "llama.cpp dir", placeholder="~/llama.cpp"),
            Field("output",    "Output GGUF path",
                  placeholder="exports/toolcalling/model-q4_k_m.gguf"),
            Field("quant",     "Quantization", placeholder="Q4_K_M", default="Q4_K_M"),
        ],
    ),
    Action(
        name="push",
        label="Push to HF Hub",
        icon="☁",
        description="Upload fused model to Hugging Face Hub (manual — you decide when).",
        fields=[
            Field("recipe",  "Recipe path",
                  placeholder="recipes/toolcalling/recipe.yaml",
                  default="recipes/toolcalling/recipe.yaml"),
            Field("repo_id", "Repo ID", placeholder="your-username/model-name"),
        ],
    ),
    Action(
        name="data_validate",
        label="Validate Data",
        icon="✅",
        description="Check every line of a JSONL file for format errors.",
        fields=[
            Field("file", "JSONL file path", placeholder="recipes/toolcalling/data/train.jsonl"),
        ],
    ),
    Action(
        name="data_convert",
        label="Convert Data",
        icon="↔",
        description="Convert Q&A pairs, CSV, or Alpaca format to chat JSONL.",
        fields=[
            Field("fmt",    "Format", placeholder="qa  |  csv  |  instruction", default="qa"),
            Field("input",  "Input file",  placeholder="my_data.csv"),
            Field("output", "Output file", placeholder="train.jsonl"),
            Field("system", "System prompt (optional)",
                  placeholder="You are a helpful assistant."),
        ],
    ),
]

ACTION_MAP = {a.name: a for a in ACTIONS}


# ── Widgets ────────────────────────────────────────────────────────────────────

class Sidebar(Widget):
    DEFAULT_CSS = """
    Sidebar {
        width: 26;
        border-right: solid $surface;
        background: $panel;
        padding: 1 0;
    }
    Sidebar ListView {
        background: $panel;
        border: none;
        height: 1fr;
        padding: 0;
    }
    Sidebar ListItem {
        padding: 0 2;
        color: $text-muted;
    }
    Sidebar ListItem.--highlight {
        background: $primary 20%;
        color: $text;
    }
    Sidebar Label.sidebar-header {
        color: $text-muted;
        padding: 0 2 1 2;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("ACTIONS", classes="sidebar-header")
        yield ListView(
            *[
                ListItem(Label(f"{action.icon}  {action.label}"), id=f"action-{action.name}")
                for action in ACTIONS
            ],
            id="action-list",
        )


class FormPanel(Widget):
    DEFAULT_CSS = """
    FormPanel {
        width: 1fr;
        border: none;
        padding: 1 3 0 3;
        height: auto;
        max-height: 40;
    }
    FormPanel Label.form-title {
        text-style: bold;
        color: $text;
    }
    FormPanel Label.form-desc {
        color: $text-muted;
        margin-bottom: 1;
    }
    FormPanel #form-fields {
        height: auto;
        overflow-y: auto;
    }
    FormPanel Label.field-label {
        color: $text-muted;
        margin-top: 1;
    }
    FormPanel Input {
        margin-bottom: 0;
    }
    FormPanel Button {
        margin-top: 1;
        margin-bottom: 1;
        width: 14;
    }
    """

    current_action: reactive[str] = reactive("download")

    def compose(self) -> ComposeResult:
        yield Label("", id="form-title", classes="form-title")
        yield Label("", id="form-desc",  classes="form-desc")
        yield ScrollableContainer(id="form-fields")
        yield Button("▶  Run", id="run-btn", variant="primary")

    def show_action(self, action: Action) -> None:
        self.current_action = action.name
        self.query_one("#form-title", Label).update(f"{action.icon}  {action.label}")
        self.query_one("#form-desc",  Label).update(action.description)

        container = self.query_one("#form-fields", ScrollableContainer)
        container.remove_children()
        for f in action.fields:
            container.mount(Label(f.label, classes="field-label"))
            container.mount(
                Input(
                    value=f.default,
                    placeholder=f.placeholder,
                    password=f.password,
                    id=f"field-{f.id}",
                )
            )

    def get_values(self) -> dict[str, str]:
        action = ACTION_MAP[self.current_action]
        result: dict[str, str] = {}
        for f in action.fields:
            try:
                result[f.id] = self.query_one(f"#field-{f.id}", Input).value.strip()
            except Exception:
                result[f.id] = f.default
        return result


class OutputPanel(Widget):
    DEFAULT_CSS = """
    OutputPanel {
        height: 1fr;
        border-top: solid $surface;
        padding: 0 1;
    }
    OutputPanel Label.out-header {
        color: $text-muted;
        padding: 0 1;
    }
    OutputPanel RichLog {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("OUTPUT", classes="out-header")
        yield RichLog(highlight=False, markup=True, id="output-log", max_lines=500)

    def write(self, text: str) -> None:
        self.query_one("#output-log", RichLog).write(text)

    def clear(self) -> None:
        self.query_one("#output-log", RichLog).clear()


# ── Messages ───────────────────────────────────────────────────────────────────

class OutputLine(Message):
    def __init__(self, text: str, style: str = "") -> None:
        super().__init__()
        self.text = text
        self.style = style


# ── Main App ───────────────────────────────────────────────────────────────────

class ForgeApp(App):
    TITLE = "mlx-forge"
    SUB_TITLE = "Mac-native fine-tuning factory"

    CSS = """
    Screen { layout: vertical; }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #right {
        layout: vertical;
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c",     "quit",         "Quit"),
        Binding("ctrl+k",     "clear_output", "Clear"),
        Binding("escape",     "focus_sidebar","Sidebar"),
        Binding("ctrl+g",     "run_action",   "Run"),
        Binding("f5",         "run_action",   "Run", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield Sidebar()
            with Vertical(id="right"):
                yield FormPanel()
                yield OutputPanel()
        yield Footer()

    def on_mount(self) -> None:
        # Select first action by default
        self.query_one("#action-list", ListView).index = 0
        self.query_one(FormPanel).show_action(ACTIONS[0])
        self.query_one("#action-list", ListView).focus()

    @on(ListView.Selected, "#action-list")
    def on_action_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("action-"):
            name = item_id[len("action-"):]
            if name in ACTION_MAP:
                self.query_one(FormPanel).show_action(ACTION_MAP[name])

    @on(Button.Pressed, "#run-btn")
    def on_run_pressed(self) -> None:
        self.action_run_action()

    def action_run_action(self) -> None:
        panel = self.query_one(FormPanel)
        name   = panel.current_action
        values = panel.get_values()
        out    = self.query_one(OutputPanel)
        out.clear()
        self._dispatch(name, values)

    def action_focus_sidebar(self) -> None:
        self.query_one("#action-list", ListView).focus()

    def action_clear_output(self) -> None:
        self.query_one(OutputPanel).clear()

    @on(OutputLine)
    def on_output_line(self, msg: OutputLine) -> None:
        text = f"[{msg.style}]{msg.text}[/]" if msg.style else msg.text
        self.query_one(OutputPanel).write(text)

    def _out(self, text: str, style: str = "") -> None:
        self.call_from_thread(self.post_message, OutputLine(text, style))

    def _dispatch(self, name: str, values: dict[str, str]) -> None:
        handlers = {
            "download":     self._run_download,
            "train":        self._run_train,
            "eval":         self._run_eval,
            "fuse":         self._run_fuse,
            "loop":         self._run_loop,
            "export":       self._run_export,
            "push":         self._run_push,
            "data_validate":self._run_data_validate,
            "data_convert": self._run_data_convert,
        }
        handler = handlers.get(name)
        if handler:
            handler(values)

    # ── Workers ────────────────────────────────────────────────────────────────

    def _stream(self, cmd: list[str], prefix: str = "") -> int:
        """Run a subprocess and stream stdout to the output panel."""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for raw in proc.stdout:
            self._out((prefix + raw).rstrip())
        proc.wait()
        return proc.returncode

    @work(thread=True)
    def _run_download(self, v: dict) -> None:
        model_id = v.get("model_id", "").strip()
        if not model_id:
            self._out("Model ID is required.", "red")
            return
        self._out(f"Downloading [cyan]{model_id}[/] …")
        try:
            from huggingface_hub import snapshot_download
            local = snapshot_download(
                repo_id=model_id,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
            )
            self._out(f"Loading model to verify …")
            from mlx_lm import load
            load(model_id)
            self._out(f"[green]✓ {model_id}[/]")
            self._out(f"  path: {local}", "dim")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_train(self, v: dict) -> None:
        recipe = v.get("recipe", "")
        if not recipe:
            self._out("Recipe path is required.", "red")
            return
        try:
            from core.config import load_recipe
            import dataclasses
            cfg = load_recipe(recipe)
            if v.get("iters"):  cfg = dataclasses.replace(cfg, iters=int(v["iters"]))
            if v.get("rank"):   cfg = dataclasses.replace(cfg, lora_rank=int(v["rank"]))
            from core.train import build_train_command
            self._out(f"Training [cyan]{Path(cfg.base_model).name}[/] for {cfg.iters} iters …")
            rc = self._stream(build_train_command(cfg))
            if rc == 0:
                self._out(f"[green]✓ adapter saved to {cfg.adapter_path}[/]")
            else:
                self._out(f"[red]Training failed (exit {rc})[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_eval(self, v: dict) -> None:
        recipe = v.get("recipe", "")
        if not recipe:
            self._out("Recipe path is required.", "red")
            return
        try:
            import importlib
            from core.config import load_recipe
            cfg  = load_recipe(recipe)
            name = Path(recipe).parent.name
            self._out(f"Evaluating [cyan]{name}[/] …")
            mod   = importlib.import_module(f"recipes.{name}.eval")
            score = mod.evaluate(cfg.base_model, cfg.data_dir, adapter_path=cfg.adapter_path)
            color = "green" if score >= 0.85 else "yellow" if score >= 0.70 else "red"
            self._out(f"score = [{color}]{score:.4f}[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_fuse(self, v: dict) -> None:
        recipe = v.get("recipe", "")
        if not recipe:
            self._out("Recipe path is required.", "red")
            return
        try:
            from core.config import load_recipe
            from core.fuse import fuse_adapter
            cfg = load_recipe(recipe)
            self._out(f"Fusing adapter into [cyan]{Path(cfg.base_model).name}[/] …")
            fuse_adapter(cfg)
            self._out(f"[green]✓ fused weights saved to {cfg.fused_path}[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_loop(self, v: dict) -> None:
        recipe = v.get("recipe", "")
        if not recipe:
            self._out("Recipe path is required.", "red")
            return
        try:
            n      = int(v.get("n_experiments") or 10)
            target = float(v.get("target_score") or 0.90)
            from core.loop import ratchet_loop
            self._out(f"Starting ratchet loop: {n} experiments, target={target} …")
            ratchet_loop(recipe, n_experiments=n, target_score=target)
            self._out("[green]✓ Loop complete.[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_export(self, v: dict) -> None:
        recipe = v.get("recipe", "")
        if not recipe or not v.get("llama_cpp") or not v.get("output"):
            self._out("Recipe, llama.cpp dir, and output path are all required.", "red")
            return
        try:
            from core.config import load_recipe
            from core.export_gguf import export_gguf
            cfg = load_recipe(recipe)
            self._out(f"Exporting to GGUF ({v.get('quant', 'Q4_K_M')}) …")
            export_gguf(
                fused_path=cfg.fused_path,
                output_gguf=v["output"],
                llama_cpp_dir=v["llama_cpp"],
                quantization=v.get("quant", "Q4_K_M"),
                system_prompt=None,
            )
            self._out(f"[green]✓ {v['output']}[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_push(self, v: dict) -> None:
        recipe = v.get("recipe", "")
        repo   = v.get("repo_id", "")
        if not recipe or not repo:
            self._out("Recipe and repo ID are both required.", "red")
            return
        try:
            from core.config import load_recipe
            from core.push_hf import push_to_hf
            cfg = load_recipe(recipe)
            self._out(f"Pushing to [cyan]huggingface.co/{repo}[/] …")
            push_to_hf(cfg.fused_path, repo, cfg.base_model, recipe, private=True)
            self._out(f"[green]✓ huggingface.co/{repo}[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_data_validate(self, v: dict) -> None:
        path = v.get("file", "")
        if not path:
            self._out("File path is required.", "red")
            return
        try:
            from core.datakit import validate_jsonl
            self._out(f"Validating [cyan]{path}[/] …")
            report = validate_jsonl(path)
            self._out(report.summary())
            if report.is_clean:
                self._out("[green]✓ All lines valid.[/]")
            else:
                self._out(f"[red]{report.error_count} error(s) found.[/]")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")

    @work(thread=True)
    def _run_data_convert(self, v: dict) -> None:
        fmt    = v.get("fmt", "qa")
        inp    = v.get("input", "")
        out    = v.get("output", "")
        system = v.get("system") or None
        if not inp or not out:
            self._out("Input and output paths are required.", "red")
            return
        try:
            from core.datakit import convert_qa_pairs, convert_csv, convert_instruction_pairs, save_jsonl
            import json
            self._out(f"Converting [cyan]{inp}[/] → [cyan]{out}[/] (format: {fmt}) …")
            if fmt == "csv":
                result = convert_csv(inp, "input", "output", system)
            elif fmt == "instruction":
                raw = [json.loads(l) for l in Path(inp).read_text().splitlines() if l.strip()]
                result = convert_instruction_pairs(raw, system)
            else:
                raw = [json.loads(l) for l in Path(inp).read_text().splitlines() if l.strip()]
                result = convert_qa_pairs(raw, system)
            if result.errors:
                for e in result.errors:
                    self._out(f"[red]error:[/] {e}")
            else:
                save_jsonl(result.examples, out)
                self._out(f"[green]✓ {result.count} examples written to {out}[/]")
                if result.skipped:
                    self._out(f"  {result.skipped} rows skipped (empty input or output)", "dim")
        except Exception as exc:
            self._out(f"[red]Error:[/] {exc}")


# ── Single-run training TUI (for mlx-forge train with live progress bar) ───────

class TrainApp(App):
    TITLE = "mlx-forge train"
    CSS = """
    Screen { layout: vertical; }
    #info { height: 1; background: $panel; padding: 0 2; color: $text-muted; }
    #prog { height: 3; padding: 1 2; }
    #log  { height: 1fr; border: round $primary; padding: 0 1; }
    """
    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="info")
        yield ProgressBar(total=self.cfg.iters, show_eta=True, id="prog")
        yield RichLog(highlight=False, markup=True, id="log", max_lines=500)
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = Path(self.cfg.adapter_path).name
        self._update_info(0, 0.0)
        self._run()

    def _update_info(self, it: int, loss: float) -> None:
        self.query_one("#info", Static).update(
            f"model: [white]{Path(self.cfg.base_model).name}[/]  "
            f"iter: [white]{it}/{self.cfg.iters}[/]  "
            + (f"loss: [bold white]{loss:.4f}[/]" if loss else "")
        )

    @work(thread=True)
    def _run(self) -> None:
        from core.train import build_train_command
        cmd  = build_train_command(self.cfg)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            self.call_from_thread(self.query_one("#log", RichLog).write, f"[dim]{line}[/]")
            m = _ITER_RE.search(line)
            if m:
                it, loss = int(m.group(1)), float(m.group(2))
                self.call_from_thread(setattr, self.query_one("#prog", ProgressBar), "progress", it)
                self.call_from_thread(self._update_info, it, loss)
        proc.wait()
        rc = proc.returncode
        self.call_from_thread(
            self.query_one("#log", RichLog).write,
            "[green]✓ Training complete.[/]" if rc == 0 else f"[red]✗ Failed (exit {rc})[/]",
        )


# ── Entry points ───────────────────────────────────────────────────────────────

def run_interactive() -> None:
    """Launch the full interactive TUI."""
    ForgeApp().run()


def run_train_tui(cfg: Any) -> None:
    """Launch the single-run training dashboard."""
    TrainApp(cfg).run()


def run_loop_tui(recipe_path: str, n_experiments: int = 10,
                 target_score: float = 0.90, seed: int = 42) -> None:
    """Launch the interactive TUI with loop pre-selected."""
    app = ForgeApp()
    # Pre-select the loop action after mount
    app.run()
