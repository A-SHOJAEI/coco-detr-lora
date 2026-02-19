from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from coco_detr_lora.utils.io import read_json, write_text


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    obj = read_json(args.results)
    res = obj["results"]

    lines: list[str] = []
    lines.append("# coco-detr-lora report")
    lines.append("")
    lines.append(f"Config: `{obj.get('config','')}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| run | AP | AP50 | AP75 | AR100 | imgs/s | peak_vram_mb | ece | nll | trainable_params |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in res:
        m = r["metrics"]
        coco = m["coco"]
        thr = m["throughput"]
        calib = m["calibration"]
        params = m["params"]
        lines.append(
            "| {run} | {AP} | {AP50} | {AP75} | {AR100} | {ips} | {vram} | {ece} | {nll} | {trainable} |".format(
                run=r["name"],
                AP=_fmt(coco["AP"]),
                AP50=_fmt(coco["AP50"]),
                AP75=_fmt(coco["AP75"]),
                AR100=_fmt(coco["AR100"]),
                ips=_fmt(thr["images_per_s"]),
                vram=_fmt(thr["peak_vram_bytes"] / (1024 * 1024)),
                ece=_fmt(calib["ece"]),
                nll=_fmt(calib["nll"]),
                trainable=_fmt(params["trainable"]),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Calibration is computed on per-detection correctness: a prediction is correct if it matches an unmatched GT of the same class with IoU >= match_iou (greedy by score).")
    lines.append("- NLL is binary cross-entropy on that correctness label using the detection confidence as probability.")
    lines.append("")

    write_text(args.out, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

