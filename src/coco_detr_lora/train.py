from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm

from coco_detr_lora.config import save_config
from coco_detr_lora.data.datasets import CocoDetectionAdapter, CocoPaths, collate_fn, make_random_hflip
from coco_detr_lora.data.synthetic_coco import SyntheticCocoSpec, generate_synthetic_coco
from coco_detr_lora.models.build import build_model, count_parameters
from coco_detr_lora.models.trainability import apply_finetune_regime
from coco_detr_lora.utils.dist import DistInfo, barrier, destroy_distributed, get_dist_info, init_distributed
from coco_detr_lora.utils.io import ensure_dir
from coco_detr_lora.utils.repro import Timer, configure_determinism, dataloader_worker_init_fn, seed_everything


def _auto_device(device_cfg: str, local_rank: int) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda", local_rank if torch.cuda.is_available() else 0)
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _sum_loss(loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for v in loss_dict.values():
        total = v if total is None else total + v
    if total is None:
        return torch.tensor(0.0)
    return total


def _save_checkpoint(out_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, cfg: dict[str, Any]) -> None:
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    path = ckpt_dir / "last.pt"
    obj = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(obj, path)


def _load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    obj = torch.load(path, map_location="cpu")
    model.load_state_dict(obj["model"], strict=True)
    if optimizer is not None and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    return int(obj.get("step", 0))


def run_training(cfg: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    project = cfg["project"]
    seed = int(project["seed"])
    deterministic = bool(project.get("deterministic", False))
    seed_everything(seed)
    configure_determinism(deterministic)

    runtime = cfg["runtime"]
    dist_info = get_dist_info(bool(runtime.get("ddp", False)))
    init_distributed(dist_info)

    device = _auto_device(str(runtime.get("device", "auto")), dist_info.local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_cfg = cfg["data"]
    data_root = Path(str(data_cfg["root"]))

    # Auto-generate synthetic data if configured and not already present
    if data_cfg.get("kind") == "synthetic_coco":
        spec = SyntheticCocoSpec(
            root=data_root,
            image_size=(int(data_cfg["image_size"][0]), int(data_cfg["image_size"][1])),
            num_classes=int(data_cfg["num_classes"]),
            train_images=int(data_cfg["train_images"]),
            val_images=int(data_cfg["val_images"]),
            seed=seed,
        )
        generate_synthetic_coco(spec)

    coco_paths = CocoPaths(root=data_root)

    model_cfg = exp["model"]
    built = build_model(model_cfg)
    model = built.model

    apply_finetune_regime(model, arch=model_cfg["arch"], regime=str(model_cfg.get("regime", "full")))

    model.to(device)
    if dist_info.enabled:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    # Dataset label_mode must match model.
    label_mode = built.label_mode
    train_tfm = make_random_hflip(0.5)
    train_ds = CocoDetectionAdapter(coco_paths.train_images, coco_paths.train_ann, label_mode=label_mode, transforms=train_tfm)

    num_workers = int(runtime.get("num_workers", 4))
    train_bs = int(cfg["train"]["batch_size"])

    if dist_info.enabled:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
    else:
        train_sampler = RandomSampler(train_ds)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        worker_init_fn=dataloader_worker_init_fn,
        generator=g,
    )

    optimizer = _make_optimizer(model, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    out_dir = Path(str(exp["out_dir"]))
    if dist_info.is_main:
        ensure_dir(out_dir)
        save_config(cfg, out_dir / "resolved_config.yaml")

    # AMP is enabled only on CUDA.
    amp_enabled = bool(runtime.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    max_steps = cfg["train"].get("max_steps", None)
    # Treat max_steps=0 or max_steps=null as "no step limit" (fall through to epoch-based training)
    max_steps_i = int(max_steps) if max_steps is not None and int(max_steps) > 0 else None
    epochs = cfg["train"].get("epochs", None)
    epochs_i = int(epochs) if epochs is not None else 1

    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 1))
    timer = Timer.start_now()

    # Reset VRAM peak tracking.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    global_step = 0
    total = max_steps_i if max_steps_i is not None else epochs_i * len(train_loader)
    pbar = tqdm(total=total, disable=not dist_info.is_main, desc=f"train:{exp['name']}")

    def _train_step(images: list[torch.Tensor], targets: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        nonlocal global_step
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss_dict = model(images, targets)  # type: ignore[operator]
            if not isinstance(loss_dict, dict):
                raise RuntimeError(f"expected dict loss output, got {type(loss_dict)}")
            loss = _sum_loss(loss_dict)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if dist_info.is_main and (global_step % log_every == 0):
            elapsed = timer.elapsed_s()
            step_log = {
                "step": global_step,
                "loss": float(loss.detach().cpu().item()),
                "losses": {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()},
                "elapsed_s": elapsed,
            }
            log_path = out_dir / "train_log.jsonl"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(step_log) + "\n")

        global_step += 1
        if dist_info.is_main:
            pbar.update(1)
        return loss, loss_dict

    if max_steps_i is not None:
        epoch = 0
        while global_step < max_steps_i:
            if dist_info.enabled and isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            for images, targets in train_loader:
                if global_step >= max_steps_i:
                    break
                _train_step(images, targets)
            epoch += 1
    else:
        for epoch in range(epochs_i):
            if dist_info.enabled and isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            for images, targets in train_loader:
                _train_step(images, targets)
            if dist_info.is_main and save_every > 0 and ((epoch + 1) % save_every == 0):
                base_model = model.module if isinstance(model, DDP) else model
                _save_checkpoint(out_dir, base_model, optimizer, global_step, cfg)

    if dist_info.is_main:
        pbar.close()

    # Save checkpoint on main process only.
    if dist_info.is_main:
        base_model = model.module if isinstance(model, DDP) else model
        _save_checkpoint(out_dir, base_model, optimizer, global_step, cfg)

    barrier(dist_info)

    # Return metadata for evaluation/reporting.
    base_model = model.module if isinstance(model, DDP) else model
    params = count_parameters(base_model)
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    destroy_distributed(dist_info)

    return {
        "steps": global_step,
        "elapsed_s": float(timer.elapsed_s()),
        "params": params,
        "peak_vram_bytes": peak_vram,
    }
