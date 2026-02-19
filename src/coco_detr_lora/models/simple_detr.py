from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class _ResNet50Backbone(nn.Module):
    """
    Minimal ResNet-50 backbone wrapper that exposes `body.layer4` like torchvision detection backbones.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = torchvision.models.resnet50(weights=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.body
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return x


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (x1 - x0).clamp(min=0)
    h = (y1 - y0).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


class SimpleDETR(nn.Module):
    """
    A small DETR-like model intended for repo smoke runs and environments where torchvision DETR
    isn't available. It provides the same top-level attributes expected by the project's LoRA and
    trainability code: `backbone.body.layer4`, `class_embed`, `bbox_embed`, `query_embed`.

    Notes:
    - Loss is a lightweight placeholder (first-K matching) suitable for smoke training; it is not
      a full DETR Hungarian-matching implementation.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_queries: int = 50,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_pos: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)

        self.backbone = _ResNet50Backbone()
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)

        # Learnable 2D positional embeddings (works for typical <=256/32 feature sizes).
        self.row_embed = nn.Embedding(max_pos, d_model // 2)
        self.col_embed = nn.Embedding(max_pos, d_model // 2)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for "no-object"

        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def _pos_embed(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if h > self.row_embed.num_embeddings or w > self.col_embed.num_embeddings:
            raise ValueError(f"feature map too large for positional embedding: h={h} w={w}")
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)  # (w, d/2)
        y_emb = self.row_embed(j)  # (h, d/2)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        )  # (h, w, d)
        return pos

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None) -> Any:
        if not images:
            raise ValueError("expected non-empty images list")

        # This repo uses fixed-size synthetic images for smoke; for simplicity we require equal shapes.
        h0, w0 = int(images[0].shape[1]), int(images[0].shape[2])
        if any(int(im.shape[1]) != h0 or int(im.shape[2]) != w0 for im in images):
            raise ValueError("SimpleDETR requires all images to have the same HxW")

        x = torch.stack(images, dim=0)  # (B,C,H,W)
        feats = self.backbone(x)  # (B,2048,h,w)
        src = self.input_proj(feats)  # (B,d,h,w)
        b, d, h, w = src.shape

        pos = self._pos_embed(h, w, device=src.device)  # (h,w,d)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)  # (B,d,h,w)

        src = (src + pos).flatten(2).transpose(1, 2)  # (B, hw, d)
        memory = self.transformer.encoder(src)  # (B, hw, d)

        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, nq, d)
        tgt = torch.zeros_like(q)
        hs = self.transformer.decoder(tgt + q, memory)  # (B, nq, d)

        logits = self.class_embed(hs)  # (B, nq, C+1)
        boxes_cxcywh = self.bbox_embed(hs).sigmoid()  # normalized cxcywh in [0,1]

        if targets is not None:
            # Placeholder losses for smoke training (not full DETR matching).
            loss_ce = torch.tensor(0.0, device=logits.device)
            loss_bbox = torch.tensor(0.0, device=logits.device)
            for bi, t in enumerate(targets):
                t_labels = t["labels"]
                t_boxes = t["boxes"]
                if not torch.is_tensor(t_labels) or not torch.is_tensor(t_boxes):
                    continue
                m = min(int(t_labels.numel()), self.num_queries)
                if m <= 0:
                    continue

                # labels are expected to be 0..num_classes-1 (dataset label_mode="detr")
                loss_ce = loss_ce + F.cross_entropy(logits[bi, :m, :], t_labels[:m].clamp(min=0, max=self.num_classes))

                # Normalize GT boxes to cxcywh in [0,1]
                gt_xyxy = t_boxes[:m]
                gt_cxcywh = _xyxy_to_cxcywh(gt_xyxy)
                gt_cxcywh[:, 0::2] = gt_cxcywh[:, 0::2] / float(w0)
                gt_cxcywh[:, 1::2] = gt_cxcywh[:, 1::2] / float(h0)
                loss_bbox = loss_bbox + F.l1_loss(boxes_cxcywh[bi, :m, :], gt_cxcywh, reduction="mean")

            return {"loss_ce": loss_ce, "loss_bbox": loss_bbox}

        # Inference: convert to per-image detections.
        probs = logits.softmax(-1)
        scores, labels = probs[..., : self.num_classes].max(-1)  # exclude no-object

        # Convert normalized cxcywh -> absolute xyxy.
        boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)
        boxes_xyxy[..., 0::2] = boxes_xyxy[..., 0::2] * float(w0)
        boxes_xyxy[..., 1::2] = boxes_xyxy[..., 1::2] * float(h0)
        boxes_xyxy = boxes_xyxy.clamp(min=0.0)

        out: list[dict[str, torch.Tensor]] = []
        for bi in range(b):
            out.append(
                {
                    "boxes": boxes_xyxy[bi].to(dtype=torch.float32),
                    "scores": scores[bi].to(dtype=torch.float32),
                    "labels": labels[bi].to(dtype=torch.int64),
                }
            )
        return out


def build_simple_detr_resnet50(*, num_classes: int) -> SimpleDETR:
    return SimpleDETR(num_classes=int(num_classes))

