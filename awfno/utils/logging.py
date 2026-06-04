"""
Lightweight logging utilities — CSV logger + Python logger factory.

No external deps beyond stdlib so tests and CI can run without wandb/tb.
WandB and TensorBoard are optional; they are enabled via experiment YAML.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional


def get_logger(name: str = "aw-fno", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a clean format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class CSVLogger:
    """
    Appends metric dicts to a CSV file, one row per call.

    Creates the file on first write and auto-generates headers from the
    first dict's keys.

    Usage::

        logger = CSVLogger("results/awfno_ns/metrics.csv")
        logger.log(epoch=1, train_loss=0.5, test_rel_l2=0.12)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._fieldnames: Optional[list] = None

    def log(self, **kwargs) -> None:
        if self._file is None:
            self._fieldnames = list(kwargs.keys())
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __del__(self) -> None:
        self.close()
