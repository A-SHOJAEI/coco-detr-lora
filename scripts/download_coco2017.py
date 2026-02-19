from __future__ import annotations

import argparse
from pathlib import Path

from coco_detr_lora.data.coco2017 import Coco2017Spec, prepare_coco2017


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory, e.g. data/coco")
    ap.add_argument(
        "--train-url",
        default="http://images.cocodataset.org/zips/train2017.zip",
        help="COCO train2017.zip URL",
    )
    ap.add_argument(
        "--val-url",
        default="http://images.cocodataset.org/zips/val2017.zip",
        help="COCO val2017.zip URL",
    )
    ap.add_argument(
        "--ann-url",
        default="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        help="COCO annotations_trainval2017.zip URL",
    )
    args = ap.parse_args()

    spec = Coco2017Spec(
        root=Path(args.out),
        train_url=args.train_url,
        val_url=args.val_url,
        ann_url=args.ann_url,
    )
    prepare_coco2017(spec)


if __name__ == "__main__":
    main()

