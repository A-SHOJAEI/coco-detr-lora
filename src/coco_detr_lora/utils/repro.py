from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(deterministic: bool) -> None:
    # Note: full determinism can reduce performance and may throw on unsupported ops.
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some builds/ops may not support this; keep best-effort flags above.
            pass


def dataloader_worker_init_fn(worker_id: int) -> None:
    # Make workers deterministically seeded based on the global seed set in the parent process.
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


@dataclass(frozen=True)
class Timer:
    start: float

    @staticmethod
    def start_now() -> "Timer":
        return Timer(start=time.time())

    def elapsed_s(self) -> float:
        return time.time() - self.start

