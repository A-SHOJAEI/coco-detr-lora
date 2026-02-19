from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from coco_detr_lora.data.datasets import CocoDetectionAdapter, CocoPaths, collate_fn
from coco_detr_lora.data.synthetic_coco import SyntheticCocoSpec, generate_synthetic_coco
from coco_detr_lora.models.build import build_model, count_parameters

try:
    # Optional: exact COCO metrics when pycocotools is available.
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore

    _HAVE_COCOEVAL = True
except Exception:  # pragma: no cover
    COCO = object  # type: ignore
    COCOeval = object  # type: ignore
    _HAVE_COCOEVAL = False


def _boxes_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _ece_and_nll(conf: np.ndarray, correct: np.ndarray, bins: int) -> tuple[float, float]:
    conf = np.clip(conf, 1e-8, 1 - 1e-8)
    correct = correct.astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(correct[mask]))
        avg_conf = float(np.mean(conf[mask]))
        ece += (float(np.sum(mask)) / float(n)) * abs(acc - avg_conf)

    # Binary NLL (BCE) on detection correctness.
    nll = float(np.mean(-(correct * np.log(conf) + (1.0 - correct) * np.log(1.0 - conf))))
    return float(ece), float(nll)


def _load_checkpoint(path: Path, model: torch.nn.Module) -> int:
    obj = torch.load(path, map_location="cpu")
    model.load_state_dict(obj["model"], strict=True)
    return int(obj.get("step", 0))


def _ap_101(rec: np.ndarray, prec: np.ndarray) -> float:
    # COCO-style 101-point interpolation.
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    rs = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in rs:
        inds = np.where(mrec >= r)[0]
        ap += float(np.max(mpre[inds])) if inds.size > 0 else 0.0
    return float(ap / 101.0)


def _coco_like_bbox_metrics(
    *,
    gt_by_image: dict[int, list[tuple[int, np.ndarray]]],
    pred_by_image: dict[int, list[tuple[int, float, np.ndarray]]],
    cat_ids: list[int],
    max_det: int,
) -> dict[str, Any]:
    # Approximate COCO bbox AP/AR without pycocotools. Exact COCOeval is used when available.
    iou_thrs = [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)]

    # Build per-class structures.
    gt_c: dict[int, dict[int, list[np.ndarray]]] = {cid: {} for cid in cat_ids}
    preds_c: dict[int, list[tuple[float, int, np.ndarray]]] = {cid: [] for cid in cat_ids}  # (score, image_id, box)
    total_gt = 0
    for img_id, gts in gt_by_image.items():
        for cid, bb in gts:
            if cid not in gt_c:
                continue
            gt_c[cid].setdefault(img_id, []).append(bb)
            total_gt += 1
    for img_id, preds in pred_by_image.items():
        for cid, sc, bb in preds:
            if cid not in preds_c:
                continue
            preds_c[cid].append((float(sc), int(img_id), bb))

    # AP per IoU threshold (averaged over classes with GT).
    ap_by_thr: dict[float, float] = {}
    ap50 = 0.0
    ap75 = 0.0
    for thr in iou_thrs:
        aps: list[float] = []
        for cid in cat_ids:
            # Count GT for this class.
            n_gt = sum(len(v) for v in gt_c[cid].values())
            if n_gt == 0:
                continue

            preds = sorted(preds_c[cid], key=lambda x: -x[0])
            used: dict[int, np.ndarray] = {img_id: np.zeros((len(bbs),), dtype=bool) for img_id, bbs in gt_c[cid].items()}

            tp = np.zeros((len(preds),), dtype=np.float64)
            fp = np.zeros((len(preds),), dtype=np.float64)
            for i, (_sc, img_id, bb) in enumerate(preds):
                gts = gt_c[cid].get(img_id, [])
                if not gts:
                    fp[i] = 1.0
                    continue
                u = used[img_id]
                best_iou = 0.0
                best_j = -1
                for j, gtbb in enumerate(gts):
                    if u[j]:
                        continue
                    iou = _iou_xyxy(bb, gtbb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= thr and best_j >= 0:
                    tp[i] = 1.0
                    u[best_j] = True
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / float(n_gt)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            aps.append(_ap_101(rec, prec))

        ap_t = float(np.mean(aps)) if aps else 0.0
        ap_by_thr[thr] = ap_t
        if abs(thr - 0.50) < 1e-9:
            ap50 = ap_t
        if abs(thr - 0.75) < 1e-9:
            ap75 = ap_t

    ap = float(np.mean(list(ap_by_thr.values()))) if ap_by_thr else 0.0

    # AR@K: per-image top-K across all classes, matched to GT of same class.
    def _ar_for_k(k: int) -> float:
        if total_gt == 0:
            return 0.0
        recalls: list[float] = []
        for thr in iou_thrs:
            matched = 0
            for img_id, gts in gt_by_image.items():
                preds = sorted(pred_by_image.get(img_id, []), key=lambda x: -x[1])[:k]
                gt_local: dict[int, list[np.ndarray]] = {}
                for cid, bb in gts:
                    gt_local.setdefault(cid, []).append(bb)
                used_local: dict[int, np.ndarray] = {cid: np.zeros((len(bbs),), dtype=bool) for cid, bbs in gt_local.items()}
                for cid, _sc, bb in preds:
                    gts_cid = gt_local.get(cid, [])
                    if not gts_cid:
                        continue
                    u = used_local[cid]
                    best_iou = 0.0
                    best_j = -1
                    for j, gtbb in enumerate(gts_cid):
                        if u[j]:
                            continue
                        iou = _iou_xyxy(bb, gtbb)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_iou >= thr and best_j >= 0:
                        u[best_j] = True
                        matched += 1
            recalls.append(float(matched / float(total_gt)))
        return float(np.mean(recalls)) if recalls else 0.0

    ar1 = _ar_for_k(1)
    ar10 = _ar_for_k(10)
    ar100 = _ar_for_k(min(100, max_det))

    nan = float("nan")
    return {
        "AP": float(ap),
        "AP50": float(ap50),
        "AP75": float(ap75),
        "APs": nan,
        "APm": nan,
        "APl": nan,
        "AR1": float(ar1),
        "AR10": float(ar10),
        "AR100": float(ar100),
        "ARs": nan,
        "ARm": nan,
        "ARl": nan,
        "impl": "coco_like_python",
    }


def evaluate_run(cfg: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg["runtime"]
    device_cfg = str(runtime.get("device", "auto"))
    device = torch.device("cuda") if (device_cfg != "cpu" and torch.cuda.is_available()) else torch.device("cpu")

    data_root = Path(str(cfg["data"]["root"]))

    # Auto-generate synthetic data if configured and not already present
    data_cfg = cfg["data"]
    if data_cfg.get("kind") == "synthetic_coco":
        spec = SyntheticCocoSpec(
            root=data_root,
            image_size=(int(data_cfg["image_size"][0]), int(data_cfg["image_size"][1])),
            num_classes=int(data_cfg["num_classes"]),
            train_images=int(data_cfg["train_images"]),
            val_images=int(data_cfg["val_images"]),
            seed=int(cfg["project"]["seed"]),
        )
        generate_synthetic_coco(spec)

    coco_paths = CocoPaths(root=data_root)
    model_cfg = exp["model"]
    built = build_model(model_cfg)
    model = built.model
    model.eval()

    out_dir = Path(str(exp["out_dir"]))
    ckpt = out_dir / "checkpoints" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    step = _load_checkpoint(ckpt, model)

    model.to(device)
    params = count_parameters(model)

    val_ds = CocoDetectionAdapter(coco_paths.val_images, coco_paths.val_ann, label_mode=built.label_mode, transforms=None)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=SequentialSampler(val_ds),
        num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    dets: list[dict[str, Any]] = []
    gt_by_image: dict[int, list[tuple[int, np.ndarray]]] = {}
    pred_by_image: dict[int, list[tuple[int, float, np.ndarray]]] = {}
    valid_cat_ids: set[int] = set()

    conf_list: list[float] = []
    correct_list: list[int] = []

    score_thr = float(cfg["eval"].get("score_threshold", 0.05))
    max_det = int(cfg["eval"].get("max_detections_per_image", 100))
    bins = int(cfg["eval"].get("calib_bins", 15))
    match_iou = float(cfg["eval"].get("match_iou", 0.5))

    # Reset peak memory before eval.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"eval:{exp['name']}", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)  # type: ignore[operator]
            if not isinstance(outputs, list) or not outputs:
                raise RuntimeError("expected list of detections")
            out = outputs[0]

            image_id = int(targets[0]["image_id"].item())

            # Ground-truth (category_id space).
            gt_boxes = targets[0]["boxes"].detach().cpu().numpy().astype(np.float64)
            gt_labels = targets[0]["labels"].detach().cpu().numpy().astype(np.int64)
            if built.label_mode == "detr":
                gt_cat_ids = gt_labels + 1
            else:
                gt_cat_ids = gt_labels
            gts_img: list[tuple[int, np.ndarray]] = []
            for cid, bb in zip(gt_cat_ids.tolist(), gt_boxes):
                cid_i = int(cid)
                valid_cat_ids.add(cid_i)
                gts_img.append((cid_i, bb))
            gt_by_image[image_id] = gts_img

            # Predictions.
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            order = np.argsort(-scores)
            boxes = boxes[order][:max_det]
            scores = scores[order][:max_det]
            labels = labels[order][:max_det]

            keep = scores >= score_thr
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Map labels back to COCO category_id space.
            if built.label_mode == "detr":
                cat_ids = labels + 1
            else:
                cat_ids = labels

            preds_img: list[tuple[int, float, np.ndarray]] = []
            for bb, sc, cid in zip(boxes.astype(np.float64), scores.astype(np.float64), cat_ids.astype(np.int64)):
                cid_i = int(cid)
                preds_img.append((cid_i, float(sc), bb))
            pred_by_image[image_id] = preds_img

            # If COCOeval is available, also build detections in its expected JSON format.
            if _HAVE_COCOEVAL and boxes.size > 0:
                b_xywh = _boxes_xyxy_to_xywh(boxes.astype(np.float64))
                for bb, sc, cid in zip(b_xywh, scores, cat_ids):
                    dets.append(
                        {
                            "image_id": image_id,
                            "category_id": int(cid),
                            "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                            "score": float(sc),
                        }
                    )

            # Calibration: per-prediction correctness by greedy matching to GT of same class with IoU>=match_iou.
            gt_by_class: dict[int, list[np.ndarray]] = {}
            for cid, gtbb in gts_img:
                gt_by_class.setdefault(int(cid), []).append(gtbb)
            used: dict[int, set[int]] = {cid: set() for cid in gt_by_class.keys()}

            # Greedy in score order.
            if preds_img:
                preds_sorted = sorted(preds_img, key=lambda x: -x[1])
                for cid, sc, bb_xyxy in preds_sorted:
                    gts = gt_by_class.get(int(cid), [])
                    best_iou = 0.0
                    best_j = -1
                    for j, gtbb in enumerate(gts):
                        if j in used[int(cid)]:
                            continue
                        iou = _iou_xyxy(bb_xyxy, gtbb)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    is_correct = 1 if (best_iou >= match_iou and best_j >= 0) else 0
                    if is_correct:
                        used[int(cid)].add(best_j)
                    conf_list.append(float(sc))
                    correct_list.append(int(is_correct))

    elapsed = time.time() - t0
    images_per_s = float(len(val_ds) / elapsed) if elapsed > 0 else 0.0
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    # Restrict predictions to categories present in the GT split (matches COCOeval's behavior).
    valid_set = set(valid_cat_ids)
    if valid_set:
        pred_by_image = {img_id: [p for p in preds if p[0] in valid_set] for img_id, preds in pred_by_image.items()}
        if dets:
            dets = [d for d in dets if int(d.get("category_id", -1)) in valid_set]

    # Metrics
    if _HAVE_COCOEVAL:
        if len(dets) == 0:
            raise RuntimeError("no detections produced (try lowering score_threshold)")
        coco_gt = COCO(str(coco_paths.val_ann))
        coco_dt = coco_gt.loadRes(dets)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats.tolist()  # 12-element vector
        coco_metrics = {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "APs": float(stats[3]),
            "APm": float(stats[4]),
            "APl": float(stats[5]),
            "AR1": float(stats[6]),
            "AR10": float(stats[7]),
            "AR100": float(stats[8]),
            "ARs": float(stats[9]),
            "ARm": float(stats[10]),
            "ARl": float(stats[11]),
            "impl": "pycocotools",
        }
    else:
        cats = sorted(valid_cat_ids)
        coco_metrics = _coco_like_bbox_metrics(
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            cat_ids=cats,
            max_det=max_det,
        )

    if conf_list:
        ece, nll = _ece_and_nll(np.array(conf_list, dtype=np.float64), np.array(correct_list, dtype=np.int64), bins=bins)
    else:
        ece, nll = float("nan"), float("nan")

    return {
        "step": int(step),
        "params": params,
        "coco": coco_metrics,
        "throughput": {"images_per_s": float(images_per_s), "elapsed_s": float(elapsed), "peak_vram_bytes": peak_vram},
        "calibration": {"ece": float(ece), "nll": float(nll), "bins": int(bins), "match_iou": float(match_iou)},
    }
