from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def _set_all_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def _enable_modules(mods: Iterable[nn.Module]) -> None:
    for m in mods:
        for p in m.parameters(recurse=True):
            p.requires_grad_(True)


def _find_lora_params(model: nn.Module) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    for n, p in model.named_parameters():
        if "lora_" in n or n.endswith("_A") or n.endswith("_B") or n.endswith(".q_A") or n.endswith(".q_B"):
            if p.requires_grad:
                out.append(p)
    return out


def apply_finetune_regime(model: nn.Module, arch: str, regime: str) -> None:
    """
    regime (as per project plan ablations):
      - "full": full fine-tune
      - "lora_only": freeze all non-LoRA weights except DETR class/box heads (and query embeddings)
      - "partial": unfreeze backbone stage4 + heads, no LoRA
    """

    if regime not in ("full", "lora_only", "partial"):
        raise ValueError(f"unknown regime: {regime}")

    if regime == "full":
        _set_all_requires_grad(model, True)
        return

    if arch != "detr_r50":
        # Ablation regimes are defined for DETR in the plan.
        _set_all_requires_grad(model, True)
        return

    if regime == "lora_only":
        _set_all_requires_grad(model, False)

        # Enable LoRA params.
        for n, p in model.named_parameters():
            if "lora_" in n or n.endswith(("_A", "_B")):
                p.requires_grad_(True)

        # Keep detection heads trainable (plan requirement).
        heads = []
        for attr in ("class_embed", "bbox_embed", "query_embed"):
            if hasattr(model, attr):
                heads.append(getattr(model, attr))
        _enable_modules(heads)
        return

    if regime == "partial":
        _set_all_requires_grad(model, False)

        # Unfreeze backbone stage4 (ResNet layer4) and detection heads, no LoRA.
        if hasattr(model, "backbone") and hasattr(model.backbone, "body") and hasattr(model.backbone.body, "layer4"):
            _enable_modules([model.backbone.body.layer4])

        heads = []
        for attr in ("class_embed", "bbox_embed", "query_embed"):
            if hasattr(model, attr):
                heads.append(getattr(model, attr))
        _enable_modules(heads)
        return

