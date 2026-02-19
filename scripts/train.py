from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from coco_detr_lora.config import load_config
from coco_detr_lora.train import run_training
from coco_detr_lora.utils.io import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    # Default to smoke config for smoke tests, full COCO for real training
    default_config = "configs/smoke.yaml" if os.environ.get("SMOKE_TEST") else "configs/coco2017_full.yaml"
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--experiment", default=None, help="Optional experiment name to run (otherwise runs all).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exps: list[dict[str, Any]] = cfg["experiments"]
    if args.experiment is not None:
        exps = [e for e in exps if str(e.get("name")) == str(args.experiment)]
        if not exps:
            raise SystemExit(f"no experiment named {args.experiment!r} in {args.config}")

    ensure_dir("runs")
    # DDP safety: multi-experiment runs under a single torchrun can be fragile due to repeated
    # init/destroy cycles. Require one experiment per invocation when DDP is active.
    if bool(cfg.get("runtime", {}).get("ddp", False)):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1 and len(exps) != 1:
            raise SystemExit("DDP is enabled; run exactly one experiment per torchrun invocation via --experiment")

    all_results = []
    for exp in exps:
        meta = run_training(cfg, exp)
        out_dir = Path(str(exp["out_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        all_results.append({"experiment": exp.get("name", "unknown"), "meta": meta})

    # Save results in pipeline-expected format
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    training_results = {
        "experiments": all_results,
        "test_metrics": all_results[-1].get("meta", {}) if all_results else {},
    }
    (results_dir / "training_results.yaml").write_text(
        json.dumps(training_results, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
