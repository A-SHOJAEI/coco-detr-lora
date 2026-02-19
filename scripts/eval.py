from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Any

from coco_detr_lora.config import RunResult, load_config
from coco_detr_lora.eval import evaluate_run
from coco_detr_lora.utils.io import atomic_write_json, ensure_dir


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", default=None, help="Optional experiment name to evaluate (otherwise evaluates all).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exps: list[dict[str, Any]] = cfg["experiments"]
    if args.experiment is not None:
        exps = [e for e in exps if str(e.get("name")) == str(args.experiment)]
        if not exps:
            raise SystemExit(f"no experiment named {args.experiment!r} in {args.config}")

    results: list[dict[str, Any]] = []
    for exp in exps:
        metrics = evaluate_run(cfg, exp)
        rr = RunResult(
            name=str(exp["name"]),
            out_dir=str(exp["out_dir"]),
            metrics=metrics,
            meta={
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "git_commit": _git_commit(),
                "config": str(Path(args.config)),
            },
        )
        results.append(rr.to_dict())

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    atomic_write_json(
        out_path,
        {
            "project": cfg.get("project", {}),
            "config": str(Path(args.config)),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
