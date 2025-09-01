
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return Config(obj)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.raw, f, indent=2, ensure_ascii=False)

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    @property
    def excel_path(self) -> str:
        return self.raw["excel_path"]
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))
