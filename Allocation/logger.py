
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from .config import ensure_dir

def get_logger(log_dir: str, name: str = "train") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger
