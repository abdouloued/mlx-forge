from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import yaml


@dataclass
class RecipeConfig:
    base_model: str
    mode: Literal["direct", "transfer"] = "direct"
    loop_model: Optional[str] = None

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: int = 16

    iters: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 4
    grad_checkpoint: bool = True

    data_dir: str = "data"
    adapter_path: str = "adapters"
    fused_path: str = "fused"

    def validate(self) -> None:
        if self.mode not in ("direct", "transfer"):
            raise ValueError(f"mode must be 'direct' or 'transfer', got {self.mode!r}")
        if self.mode == "transfer" and self.loop_model is None:
            raise ValueError("transfer mode requires loop_model to be set")


def load_recipe(path: str | Path) -> RecipeConfig:
    data = yaml.safe_load(Path(path).read_text())
    cfg = RecipeConfig(**data)
    cfg.validate()
    return cfg
