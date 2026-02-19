from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torchvision

from coco_detr_lora.models.lora import inject_lora_into_detr_transformer


def _call_with_supported_kwargs(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


@dataclass(frozen=True)
class BuiltModel:
    model: nn.Module
    label_mode: str  # "fasterrcnn" or "detr" for dataset label mapping


def build_model(model_cfg: dict[str, Any]) -> BuiltModel:
    arch = model_cfg["arch"]
    num_classes = int(model_cfg["num_classes"])

    if arch == "fasterrcnn_r50_fpn":
        fn = torchvision.models.detection.fasterrcnn_resnet50_fpn
        # Faster R-CNN expects num_classes including background.
        model = _call_with_supported_kwargs(
            fn,
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
        return BuiltModel(model=model, label_mode="fasterrcnn")

    if arch == "detr_r50":
        # Prefer torchvision DETR when available; fall back to a small local DETR-like model
        # (keeps the repo runnable on torchvision builds that ship without DETR).
        if hasattr(torchvision.models.detection, "detr_resnet50"):
            fn = torchvision.models.detection.detr_resnet50
            # DETR expects num_classes excluding "no-object"; it internally adds +1.
            model = _call_with_supported_kwargs(
                fn,
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )
        else:
            from coco_detr_lora.models.simple_detr import build_simple_detr_resnet50

            model = build_simple_detr_resnet50(num_classes=num_classes)

        lora_cfg = model_cfg.get("lora", {}) or {}
        if bool(lora_cfg.get("enabled", False)):
            inject_lora_into_detr_transformer(
                model,
                rank=int(lora_cfg["rank"]),
                alpha=int(lora_cfg["alpha"]),
                dropout=float(lora_cfg.get("dropout", 0.0)),
                placement=str(lora_cfg.get("placement", "attention_only")),
            )

        return BuiltModel(model=model, label_mode="detr")

    raise ValueError(f"unknown arch: {arch}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
