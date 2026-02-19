from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ${a.b.c} references inside YAML values.
_REF_RE = re.compile(r"\$\{([^}]+)\}")


def _get_by_path(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _resolve_refs(obj: Any, root: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    if isinstance(obj, str):
        def _sub(m: re.Match[str]) -> str:
            return str(_get_by_path(root, m.group(1)))

        # Resolve repeatedly to allow chained refs.
        prev = obj
        for _ in range(10):
            cur = _REF_RE.sub(_sub, prev)
            if cur == prev:
                break
            prev = cur
        return prev
    return obj


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping, got {type(raw)}")
    resolved = _resolve_refs(raw, raw)
    return resolved


def save_config(cfg: dict[str, Any], path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass(frozen=True)
class RunResult:
    name: str
    out_dir: str
    metrics: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
