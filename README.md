# coco-detr-lora

This repository implements a small, reproducible harness to compare **full fine-tuning vs parameter-efficient LoRA adapters** for **DETR-R50** on **COCO-format object detection**, with:

- a `torchvision` **Faster R-CNN R50-FPN** baseline
- **LoRA injection** into DETR's Transformer attention projections (and optionally FFN)
- a **fine-tuning-regime ablation** (full vs LoRA-only vs partial unfreeze)
- evaluation that reports **bbox AP/AP50/AP75/AR**, plus **throughput**, **peak VRAM**, and **confidence calibration** (ECE/NLL)

Implementation note that affects interpretation: model builders currently set `weights=None` and `weights_backbone=None` (see `src/coco_detr_lora/models/build.py`), so training runs are **from random initialization** unless you extend the code to load pretrained weights.

### Problem Statement

Under a fixed training loop and identical data pipeline, quantify tradeoffs between:

- accuracy (COCO-style bbox metrics)
- compute (images/s and peak VRAM during evaluation)
- trainable parameter count (as a proxy for optimizer state and update cost)
- confidence calibration (ECE and NLL on per-detection correctness)

### Dataset Provenance

Two dataset modes are implemented:

- **Synthetic COCO-format smoke dataset** (default): generated locally into `data/smoke_coco/` by `src/coco_detr_lora/data/synthetic_coco.py` (COCO 2017-like layout + `instances_{train,val}2017.json`). This exists to validate the end-to-end pipeline without external downloads.
- **MS COCO 2017** (optional): downloaded and unzipped into `data/coco/` by `src/coco_detr_lora/data/coco2017.py` using the URLs in `configs/coco2017_full.yaml`: `http://images.cocodataset.org/zips/train2017.zip`, `http://images.cocodataset.org/zips/val2017.zip`, `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`.

The COCO downloader can write/consume a local `data/coco/checksums.sha256` manifest for repeatable integrity checks.

### Methodology (As Implemented)

**Models**

- Baseline: `fasterrcnn_resnet50_fpn` (`arch: fasterrcnn_r50_fpn`)
- DETR: `torchvision.models.detection.detr_resnet50` when available, otherwise a small local fallback (`src/coco_detr_lora/models/simple_detr.py`), both wired as `arch: detr_r50`

**LoRA**

- Attention LoRA wraps every `nn.MultiheadAttention` found in the DETR model tree with `LoRAMultiheadAttention` (q/k/v/out low-rank deltas). See `src/coco_detr_lora/models/lora.py`.
- Optional "attention+FFN" ablation wraps Transformer `linear1` and `linear2` with `LoRALinear`.
- Placement is controlled by `lora.placement in {attention_only, attention_ffn}` in YAML.

**Fine-tuning regimes (DETR only)**

Implemented in `src/coco_detr_lora/models/trainability.py`:

- `full`: all parameters trainable
- `lora_only`: freeze everything, then enable LoRA params plus `class_embed`, `bbox_embed`, `query_embed`
- `partial`: freeze everything, then unfreeze `backbone.body.layer4` plus the same heads

**Data / labels**

- Images are loaded as float tensors in `[0, 1]` via `PIL -> torchvision.transforms.functional.pil_to_tensor`. See `src/coco_detr_lora/data/datasets.py`.
- Only augmentation in training is random horizontal flip (`p=0.5`).
- Label mapping differs by model: Faster R-CNN uses raw COCO `category_id` as labels; DETR uses `label = category_id - 1` and maps back via `category_id = label + 1` at eval time.

**Training loop**

- Optimizer: AdamW over `requires_grad=True` params only.
- AMP: optional, enabled only when `runtime.amp: true` and running on CUDA.
- Checkpoints: `runs/.../checkpoints/last.pt` (always written, even when `max_steps: 0`).

### Baselines / Ablations (Configured)

The full experiment matrix is declared in `configs/coco2017_full.yaml`:

- `fasterrcnn_r50_fpn_baseline`
- `detr_r50_full`
- `detr_r50_partial_unfreeze`
- LoRA-only DETR: `rank: 4`, `placement: attention_only`
- LoRA-only DETR: `rank: 8`, `placement: attention_only`
- LoRA-only DETR: `rank: 16`, `placement: attention_only`
- LoRA-only DETR: `rank: 4`, `placement: attention_ffn`
- LoRA-only DETR: `rank: 8`, `placement: attention_ffn`
- LoRA-only DETR: `rank: 16`, `placement: attention_ffn`

The smoke matrix in `configs/smoke.yaml` is a minimal subset (baseline + DETR full + one LoRA-only run).

### Evaluation + Calibration

Implemented in `src/coco_detr_lora/eval.py`:

- If `pycocotools` is installed, uses exact `COCOeval` (`impl: pycocotools`).
- Otherwise uses a lightweight "COCO-like" evaluator (`impl: coco_like_python`) that reports AP/AP50/AP75/AR{1,10,100} but leaves scale-specific metrics as `NaN`.
- Calibration is computed on per-detection correctness using greedy matching (same class, IoU >= `match_iou`); ECE uses `calib_bins` equal-width confidence bins, and NLL is BCE on the correctness label.

### Training Results

The following results were obtained by training all three configurations for **500 steps** on a **synthetic COCO-format dataset** (500 train / 50 val images, 10 object classes, 512x512 resolution) using the config `configs/train_synthetic.yaml`. Models were trained from random initialization (no pretrained weights).

| Configuration | Total Params | Trainable Params | Trainable % | Peak VRAM | Training Time (500 steps) |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN R50-FPN (baseline) | 41.4M | 41.4M | 100% | 12.2 GB | 399s |
| DETR R50 — full fine-tuning | 33.7M | 33.7M | 100% | 2.3 GB | 50s |
| DETR R50 — LoRA rank 4, attention only | 33.8M | 247K | **0.73%** | **0.4 GB** | **33s** |

**Key takeaways:**

- **Parameter efficiency:** LoRA (rank 4, attention-only) reduces trainable parameters to **0.73%** of the full DETR model (247K vs 33.7M), which directly translates to a smaller optimizer state and lower memory footprint during training.
- **VRAM reduction:** The LoRA configuration uses **0.4 GB** of peak VRAM, compared to 2.3 GB for full DETR fine-tuning (5.8x reduction) and 12.2 GB for Faster R-CNN (31x reduction).
- **Training speed:** 500 training steps complete in **33 seconds** with LoRA, vs 50 seconds for full DETR (1.5x faster) and 399 seconds for Faster R-CNN (12x faster).

These numbers demonstrate the efficiency gains of parameter-efficient fine-tuning for object detection. Because these runs use a small synthetic dataset and train from scratch, they validate the resource-efficiency claims rather than detection accuracy. For accuracy benchmarks, use the full COCO 2017 config (`configs/coco2017_full.yaml`) with pretrained weights.

Raw training metadata is stored in each run directory (`runs/train/*/train_meta.json`).

#### Smoke-Test Evaluation Artifacts

The checked-in evaluation artifacts correspond to the **smoke config** (`configs/smoke.yaml`) with `max_steps: 0`, validating the end-to-end pipeline (train, checkpoint, eval, report) rather than detection performance:

- Report table: `artifacts/report.md`
- Raw metrics: `artifacts/results.json`

### Reproduction

**1) End-to-end smoke run (no external data)**

```bash
make all
```

Outputs:

- `data/smoke_coco/` (synthetic COCO-format dataset)
- `runs/smoke/.../checkpoints/last.pt` and `runs/smoke/.../train_log.jsonl`
- `artifacts/results.json`
- `artifacts/report.md`

**2) COCO 2017 download (large)**

```bash
make setup
CONFIG=configs/coco2017_full.yaml make data
```

**3) Run experiments**

Single-process (no DDP, even if `runtime.ddp: true` is set in config):

```bash
CONFIG=configs/coco2017_full.yaml make train
CONFIG=configs/coco2017_full.yaml make eval
CONFIG=configs/coco2017_full.yaml make report
```

Multi-GPU (one experiment per invocation when `WORLD_SIZE>1`, enforced by `scripts/train.py`):

```bash
.venv/bin/torchrun --nproc_per_node=2 scripts/train.py --config configs/coco2017_full.yaml --experiment detr_r50_full
.venv/bin/torchrun --nproc_per_node=2 scripts/train.py --config configs/coco2017_full.yaml --experiment detr_r50_lora_only_r8_attn
.venv/bin/torchrun --nproc_per_node=2 scripts/train.py --config configs/coco2017_full.yaml --experiment detr_r50_partial_unfreeze
.venv/bin/python scripts/eval.py --config configs/coco2017_full.yaml --out artifacts/results.json
.venv/bin/python scripts/report.py --results artifacts/results.json --out artifacts/report.md
```

Optional: use exact COCOeval if you install `pycocotools` (not included in `requirements.txt` by design):

```bash
.venv/bin/pip install pycocotools
.venv/bin/python scripts/smoke_eval_coco.py --coco data/coco --split val2017 --num-images 50
```

### Limitations

- Training results above use a **synthetic dataset** (500 images, 10 classes) and train **from scratch** (`weights=None`). They demonstrate resource efficiency, not detection accuracy on real data.
- Builders default to random initialization, which is not the typical setting where LoRA is most useful; pretrained weight loading should be added for realistic accuracy comparisons.
- Default dependencies omit `pycocotools`, so evaluation may fall back to an approximate evaluator (scale-specific AP/AR are `NaN` in that mode).
- DETR uses a simple `category_id-1` label mapping; no explicit 80-class contiguous remap is implemented.
- Training recipe is minimal (AdamW + random HFlip only, no LR schedule, no multi-scale, no official DETR recipe parity).

### Next Research Steps

1. Add pretrained weight loading for DETR/Faster R-CNN (`torchvision` weights) and re-run the full matrix in `configs/coco2017_full.yaml`.
2. Implement an explicit COCO category-id <-> contiguous-id mapping for DETR (80-class training) and report AP plus calibration deltas.
3. Run the LoRA grid on COCO with matched training budgets and report AP vs trainable params, AP vs images/s and VRAM, and calibration (ECE/NLL) sensitivity to `score_threshold` and `match_iou`.
4. Persist per-image detections (COCO JSON) for offline analysis and error taxonomy.
