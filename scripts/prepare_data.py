from __future__ import annotations

import argparse
from pathlib import Path

from coco_detr_lora.config import load_config
from coco_detr_lora.data.coco2017 import Coco2017Spec, prepare_coco2017
from coco_detr_lora.data.synthetic_coco import SyntheticCocoSpec, generate_synthetic_coco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    kind = str(data_cfg["kind"])
    root = Path(str(data_cfg["root"]))

    if kind == "synthetic_coco":
        spec = SyntheticCocoSpec(
            root=root,
            image_size=(int(data_cfg["image_size"][0]), int(data_cfg["image_size"][1])),
            num_classes=int(data_cfg["num_classes"]),
            train_images=int(data_cfg["train_images"]),
            val_images=int(data_cfg["val_images"]),
            seed=int(cfg["project"]["seed"]),
        )
        generate_synthetic_coco(spec)
        return

    if kind == "coco2017":
        coco_cfg = data_cfg["coco"]
        urls = coco_cfg["urls"]
        sha = coco_cfg.get("sha256", {}) or {}
        spec = Coco2017Spec(
            root=root,
            train_url=str(urls[0]),
            val_url=str(urls[1]),
            ann_url=str(urls[2]),
            sha256_train=sha.get("train2017.zip"),
            sha256_val=sha.get("val2017.zip"),
            sha256_ann=sha.get("annotations_trainval2017.zip"),
        )
        prepare_coco2017(spec)
        return

    raise SystemExit(f"unknown data.kind: {kind}")


if __name__ == "__main__":
    main()

