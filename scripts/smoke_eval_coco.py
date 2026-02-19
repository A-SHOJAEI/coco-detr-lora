from __future__ import annotations

import argparse
from pathlib import Path

try:
    from pycocotools.coco import COCO  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "pycocotools is not installed. This optional script requires pycocotools; "
        "the default `make all` pipeline does not. Install pycocotools in the venv if needed.\n"
        f"Import error: {e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO root, e.g. data/coco or data/smoke_coco")
    ap.add_argument("--split", default="val2017", choices=["train2017", "val2017"])
    ap.add_argument("--num-images", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.coco)
    ann = root / "annotations" / f"instances_{args.split}.json"
    if not ann.exists():
        raise SystemExit(f"missing annotations: {ann}")

    coco = COCO(str(ann))
    img_ids = coco.getImgIds()[: args.num_images]
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)
    cats = coco.loadCats(coco.getCatIds())

    print(f"split={args.split} images={len(img_ids)} anns={len(anns)} categories={len(cats)}")


if __name__ == "__main__":
    main()
