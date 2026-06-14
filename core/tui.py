"""Textual TUI for mlx-forge — live training dashboard."""
from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    DataTable, Footer, Header, Label, ProgressBar, RichLog, Static, Rule
)
from textual import work

from core.config import RecipeConfig, load_recipe
from core.train import build_train_command

# ── Regex for parsing mlx_lm.lora output ──────────────────────────────────────
_ITER_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")


# ── Messages (worker → app) ────────────────────────────────────────────────────

class ExperimentStarted(Message):
    def __init__(self, n: int, total: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.n = n
        self.total = total
        self.config = config


class IterUpdate(Message):
    def __init__(self, iter_n: int, total: int, loss: float) -> None:
        super().__init__()
        self.iter_n = iter_n
        self.total = total
        self.loss = loss


class LogLine(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ExperimentDone(Message):
    def __init__(self, n: int, score: float, kept: bool,
                 config: dict[str, Any], best_score: float) -> None:
        super().__init__()
        self.n = n
        self.score = score
        self.kept = kept
        self.config = config
        self.best_score = best_score


class RunDone(Message):
    def __init__(self, best_score: float, best_config: dict[str, Any]) -> None:
        super().__init__()
        self.best_score = best_score
        self.best_config = best_config


# ── Widgets ────────────────────────────────────────────────────────────────────

class StatusBar(Static):
    """Top status line showing recipe + hardware."""

    DEFAULT_CSS = """
    StatusBar {
        background: $panel;
        color: $text-muted;
        padding: 0 2;
        height: 1;
    }
    """

    def update_status(self, recipe: str, model: str) -> None:
        import platform
        chip = "Apple Silicon"
        try:
            import subprocess as sp
            out = sp.check_output(
                ["system_profiler", "SPHardwareDataType"], text=True, stderr=sp.DEVNULL
            )
            for line in out.splitlines():
                if "Chip:" in line:
                    chip = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
        self.update(f"recipe: [cyan]{recipe}[/]  model: [white]{Path(model).name}[/]  {chip}")


class CurrentRun(Widget):
    """Right panel — live progress for the active experiment."""

    DEFAULT_CSS = """
    CurrentRun {
        border: round $primary;
        padding: 1 2;
        height: 100%;
    }
    CurrentRun Label { margin-bottom: 1; }
    CurrentRun ProgressBar { margin-bottom: 1; }
    """

    iter_n: reactive[int] = reactive(0)
    total: reactive[int] = reactive(500)
    loss: reactive[float] = reactive(0.0)
    first_loss: reactive[float] = reactive(0.0)

    def compose(self) -> ComposeResult:
        yield Label("", id="exp-title")
        yield Label("", id="exp-config")
        yield Label("", id="exp-iter")
        yield ProgressBar(total=100, show_eta=False, id="train-progress")
        yield Label("", id="exp-loss")
        yield Rule()
        yield RichLog(highlight=False, markup=True, id="train-log", max_lines=200)

    def start_experiment(self, n: int, total_exps: int, config: dict, total_iters: int) -> None:
        self.total = total_iters
        self.iter_n = 0
        self.first_loss = 0.0
        self.loss = 0.0
        cfg_str = "  ".join(f"{k}={v}" for k, v in config.items()
                            if k in ("learning_rate", "lora_rank", "lora_layers", "batch_size"))
        self.query_one("#exp-title", Label).update(
            f"[bold white]Experiment {n} / {total_exps}[/]"
        )
        self.query_one("#exp-config", Label).update(f"[dim]{cfg_str}[/]")
        self.query_one("#exp-iter",   Label).update("")
        self.query_one("#exp-loss",   Label).update("")
        pb = self.query_one("#train-progress", ProgressBar)
        pb.total = total_iters
        pb.progress = 0

    def on_iter_update(self, msg: IterUpdate) -> None:
        if self.first_loss == 0.0 and msg.loss > 0:
            self.first_loss = msg.loss
        self.iter_n = msg.iter_n
        self.loss = msg.loss
        pb = self.query_one("#train-progress", ProgressBar)
        pb.total = msg.total
        pb.progress = msg.iter_n
        self.query_one("#exp-iter", Label).update(
            f"[dim]iter[/] [white]{msg.iter_n}[/] [dim]/ {msg.total}[/]"
        )
        if self.first_loss > 0:
            delta = self.first_loss - msg.loss
            arrow = "[green]↓[/]" if delta > 0 else "[red]↑[/]"
            self.query_one("#exp-loss", Label).update(
                f"loss  [dim]{self.first_loss:.3f}[/] → [bold white]{msg.loss:.3f}[/]  {arrow} {abs(delta):.3f}"
            )

    def append_log(self, text: str) -> None:
        log = self.query_one("#train-log", RichLog)
        log.write(f"[dim]{text}[/]")


class ExperimentTable(Widget):
    """Left panel — history of all experiments."""

    DEFAULT_CSS = """
    ExperimentTable {
        border: round $surface;
        height: 100%;
        padding: 0 1;
    }
    ExperimentTable DataTable { height: 1fr; }
    """

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]Experiments[/]", id="table-title")
        dt = DataTable(zebra_stripes=True, show_cursor=False)
        dt.add_columns("#", "score", "Δ", "config")
        yield dt

    def add_result(self, n: int, score: float, kept: bool,
                   config: dict[str, Any], best_score: float) -> None:
        dt = self.query_one(DataTable)
        delta = score - (best_score if kept else best_score)
        if kept:
            icon  = "[green]✓[/]"
            sc    = f"[bold green]{score:.4f}[/]"
            d_str = f"[green]+{abs(score - (best_score - score if not kept else 0)):.4f}[/]"
        else:
            icon  = "[dim red]✗[/]"
            sc    = f"[dim]{score:.4f}[/]"
            d_str = f"[dim red]{score - best_score:+.4f}[/]"

        lr    = config.get("learning_rate", "")
        rank  = config.get("lora_rank", "")
        cfg_s = f"lr={lr:.0e} r={rank}"
        dt.add_row(f"{icon} {n}", sc, d_str, cfg_s)

    def add_running(self, n: int, config: dict[str, Any]) -> None:
        dt = self.query_one(DataTable)
        dt.add_row(f"[cyan]↻[/] {n}", "[cyan]running[/]", "", "")


# ── Main App ───────────────────────────────────────────────────────────────────

class ForgeApp(App):
    """mlx-forge training dashboard."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    ExperimentTable {
        width: 36;
        min-width: 30;
    }

    CurrentRun {
        width: 1fr;
    }

    #best-bar {
        background: $panel;
        color: $text-muted;
        padding: 0 2;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self, recipe_path: str, n_experiments: int = 10,
                 target_score: float = 0.90, seed: int = 42) -> None:
        super().__init__()
        self.recipe_path   = recipe_path
        self.n_experiments = n_experiments
        self.target_score  = target_score
        self.seed          = seed
        self._best_score   = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar(id="status-bar")
        with Horizontal(id="body"):
            yield ExperimentTable()
            yield CurrentRun()
        yield Static("best score: —", id="best-bar")
        yield Footer()

    def on_mount(self) -> None:
        cfg = load_recipe(self.recipe_path)
        self.query_one(StatusBar).update_status(
            Path(self.recipe_path).parent.name, cfg.base_model
        )
        self.title = "mlx-forge"
        self.sub_title = Path(self.recipe_path).parent.name
        self._run_loop()

    @work(thread=True)
    def _run_loop(self) -> None:
        import random, dataclasses, json
        from core.loop import SEARCH_SPACE, propose_config, load_state, save_state, _git_commit
        from core.config import load_recipe

        rng   = random.Random(self.seed)
        cfg   = load_recipe(self.recipe_path)
        state = load_state("loop_state.json")
        best  = state.best_score

        for i in range(1, self.n_experiments + 1):
            candidate = propose_config(cfg, SEARCH_SPACE, rng)
            exp_cfg   = dataclasses.replace(
                candidate,
                adapter_path=f"{candidate.adapter_path}_exp{i:03d}",
            )
            exp_dict = {
                k: getattr(exp_cfg, k)
                for k in ("learning_rate", "lora_rank", "lora_layers", "batch_size")
            }

            self.call_from_thread(
                self.post_message, ExperimentStarted(i, self.n_experiments, exp_dict)
            )
            self.call_from_thread(
                self.query_one(ExperimentTable).add_running, i, exp_dict
            )

            # ── Run training subprocess with streaming ──
            cmd  = build_train_command(exp_cfg)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for raw in proc.stdout:
                line = raw.rstrip()
                self.call_from_thread(self.post_message, LogLine(line))
                m = _ITER_RE.search(line)
                if m:
                    self.call_from_thread(
                        self.post_message,
                        IterUpdate(int(m.group(1)), exp_cfg.iters, float(m.group(2))),
                    )
            proc.wait()

            if proc.returncode != 0:
                self.call_from_thread(self.post_message, LogLine("[red]training failed[/]"))
                continue

            # ── Score ──
            try:
                import importlib
                recipe_name = Path(self.recipe_path).parent.name
                mod   = importlib.import_module(f"recipes.{recipe_name}.eval")
                score = mod.evaluate(
                    exp_cfg.base_model,
                    exp_cfg.data_dir,
                    adapter_path=exp_cfg.adapter_path,
                )
            except Exception as exc:
                self.call_from_thread(self.post_message, LogLine(f"[red]eval error: {exc}[/]"))
                score = 0.0

            kept = score > best
            if kept:
                best = score
                state.best_score  = best
                state.experiment  = i
                state.best_config = exp_dict
                save_state(state, "loop_state.json")
                _git_commit(
                    ["loop_state.json"],
                    f"loop exp{i:03d}: score={score:.4f}",
                )

            self._best_score = best
            self.call_from_thread(
                self.post_message,
                ExperimentDone(i, score, kept, exp_dict, best),
            )
            self.call_from_thread(
                self.query_one("#best-bar", Static).update,
                f"best score: [bold green]{best:.4f}[/]  "
                f"experiment {i}/{self.n_experiments}",
            )

            if best >= self.target_score:
                break

        self.call_from_thread(
            self.post_message,
            RunDone(best, state.best_config),
        )

    # ── Message handlers ───────────────────────────────────────────────────────

    def on_experiment_started(self, msg: ExperimentStarted) -> None:
        cfg = load_recipe(self.recipe_path)
        self.query_one(CurrentRun).start_experiment(
            msg.n, msg.total, msg.config, cfg.iters
        )

    def on_iter_update(self, msg: IterUpdate) -> None:
        self.query_one(CurrentRun).on_iter_update(msg)

    def on_log_line(self, msg: LogLine) -> None:
        self.query_one(CurrentRun).append_log(msg.text)

    def on_experiment_done(self, msg: ExperimentDone) -> None:
        self.query_one(ExperimentTable).add_result(
            msg.n, msg.score, msg.kept, msg.config, msg.best_score
        )

    def on_run_done(self, msg: RunDone) -> None:
        self.query_one("#best-bar", Static).update(
            f"[bold green]✓ Done.[/]  best score: [bold green]{msg.best_score:.4f}[/]  "
            f"config: {msg.best_config}  — press Ctrl+Q to exit"
        )

    def action_clear_log(self) -> None:
        self.query_one("#train-log", RichLog).clear()


# ── Single-run variant (mlx-forge train) ──────────────────────────────────────

class TrainApp(App):
    """Minimal TUI for a single training run."""

    CSS = """
    Screen { layout: vertical; }
    #log   { height: 1fr; border: round $primary; padding: 0 1; }
    #prog  { height: 3; padding: 1 2; }
    #info  { height: 1; background: $panel; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]

    def __init__(self, cfg: RecipeConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="info")
        yield ProgressBar(total=self.cfg.iters, show_eta=True, id="prog")
        yield RichLog(highlight=False, markup=True, id="log", max_lines=500)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "mlx-forge train"
        self.sub_title = Path(self.cfg.adapter_path).name
        self.query_one("#info", Static).update(
            f"model: [white]{Path(self.cfg.base_model).name}[/]  "
            f"iters: [white]{self.cfg.iters}[/]  "
            f"rank: [white]{self.cfg.lora_rank}[/]  "
            f"layers: [white]{self.cfg.lora_layers}[/]"
        )
        self._run_train()

    @work(thread=True)
    def _run_train(self) -> None:
        cmd  = build_train_command(self.cfg)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            self.call_from_thread(
                self.query_one("#log", RichLog).write, f"[dim]{line}[/]"
            )
            m = _ITER_RE.search(line)
            if m:
                iter_n = int(m.group(1))
                loss   = float(m.group(2))
                pb = self.query_one("#prog", ProgressBar)
                self.call_from_thread(setattr, pb, "progress", iter_n)
                self.call_from_thread(
                    self.query_one("#info", Static).update,
                    f"model: [white]{Path(self.cfg.base_model).name}[/]  "
                    f"iter: [white]{iter_n}/{self.cfg.iters}[/]  "
                    f"loss: [bold white]{loss:.4f}[/]"
                )
        proc.wait()
        rc = proc.returncode
        self.call_from_thread(
            self.query_one("#log", RichLog).write,
            "[green]✓ Training complete.[/]" if rc == 0 else f"[red]✗ Training failed (exit {rc})[/]"
        )


# ── Entry points ───────────────────────────────────────────────────────────────

def run_loop_tui(recipe_path: str, n_experiments: int = 10,
                 target_score: float = 0.90, seed: int = 42) -> None:
    ForgeApp(recipe_path, n_experiments, target_score, seed).run()


def run_train_tui(cfg: RecipeConfig) -> None:
    TrainApp(cfg).run()
