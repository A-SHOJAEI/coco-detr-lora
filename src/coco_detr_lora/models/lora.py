from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float = 0.0

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank) if self.rank > 0 else 0.0


class LoRALinear(nn.Module):
    """
    Linear layer with a frozen base weight + trainable low-rank update.

    This is used for DETR FFN ablations ("attention+FFN LoRA").
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        if cfg.rank <= 0:
            raise ValueError("rank must be > 0")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # A: (r, in), B: (out, r)
        self.lora_A = nn.Parameter(torch.zeros((cfg.rank, self.in_features), dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, cfg.rank), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # Best-effort dropout on the low-rank weight delta (keeps impl simple and deterministic-friendly).
        delta_w = self.lora_B @ self.lora_A
        if self.cfg.dropout > 0:
            delta_w = F.dropout(delta_w, p=self.cfg.dropout, training=self.training)
        y = y + F.linear(x, delta_w * self.cfg.scaling, bias=None)
        return y


class LoRAMultiheadAttention(nn.Module):
    """
    MultiheadAttention with frozen base weights and trainable LoRA deltas for Q/K/V and output projection.

    This follows the plan's "attention projections (q,k,v,out)" ablation target for torchvision DETR.
    """

    def __init__(self, base: nn.MultiheadAttention, cfg: LoRAConfig) -> None:
        super().__init__()
        if cfg.rank <= 0:
            raise ValueError("rank must be > 0")
        self.cfg = cfg
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.dropout = base.dropout
        self.batch_first = getattr(base, "batch_first", False)
        self.add_zero_attn = base.add_zero_attn
        self.kdim = base.kdim
        self.vdim = base.vdim

        # Keep the original module for parameters/state, but ensure its params are frozen.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        # For typical DETR, kdim == vdim == embed_dim and base uses in_proj_weight shape (3E, E).
        if self.base.in_proj_weight is None:
            raise ValueError("expected base.in_proj_weight to exist")
        if self.base.in_proj_weight.shape[0] != 3 * self.embed_dim or self.base.in_proj_weight.shape[1] != self.embed_dim:
            raise ValueError(f"unexpected in_proj_weight shape: {tuple(self.base.in_proj_weight.shape)}")

        # LoRA for q/k/v: each delta is (E, E)
        self.q_A = nn.Parameter(torch.zeros((cfg.rank, self.embed_dim), dtype=torch.float32))
        self.q_B = nn.Parameter(torch.zeros((self.embed_dim, cfg.rank), dtype=torch.float32))
        self.k_A = nn.Parameter(torch.zeros((cfg.rank, self.embed_dim), dtype=torch.float32))
        self.k_B = nn.Parameter(torch.zeros((self.embed_dim, cfg.rank), dtype=torch.float32))
        self.v_A = nn.Parameter(torch.zeros((cfg.rank, self.embed_dim), dtype=torch.float32))
        self.v_B = nn.Parameter(torch.zeros((self.embed_dim, cfg.rank), dtype=torch.float32))
        self.o_A = nn.Parameter(torch.zeros((cfg.rank, self.embed_dim), dtype=torch.float32))
        self.o_B = nn.Parameter(torch.zeros((self.embed_dim, cfg.rank), dtype=torch.float32))

        for A in (self.q_A, self.k_A, self.v_A, self.o_A):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in (self.q_B, self.k_B, self.v_B, self.o_B):
            nn.init.zeros_(B)

    def _delta(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        delta = B @ A
        if self.cfg.dropout > 0:
            delta = F.dropout(delta, p=self.cfg.dropout, training=self.training)
        return delta * self.cfg.scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # multi_head_attention_forward expects sequence-first unless batch_first is handled by caller.
        if self.batch_first:
            # (N, L, E) -> (L, N, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        base_w = self.base.in_proj_weight
        base_b = self.base.in_proj_bias
        E = self.embed_dim

        dq = self._delta(self.q_A, self.q_B)
        dk = self._delta(self.k_A, self.k_B)
        dv = self._delta(self.v_A, self.v_B)
        in_proj_weight = torch.cat([base_w[0:E] + dq, base_w[E : 2 * E] + dk, base_w[2 * E : 3 * E] + dv], dim=0)

        out_w = self.base.out_proj.weight + self._delta(self.o_A, self.o_B)
        out_b = self.base.out_proj.bias

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=base_b,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_w,
            out_proj_bias=out_b,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights

    def __getattr__(self, name: str):
        # nn.TransformerEncoderLayer/DecoderLayer may access internal MultiheadAttention attributes
        # (e.g. `_qkv_same_embed_dim`). Delegate unknown attributes to the wrapped base module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)


def inject_lora_into_detr_transformer(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    placement: str,
) -> None:
    """
    Replace attention modules with LoRA-wrapped variants; optionally wrap FFN linears.

    placement:
      - "attention_only": only wrap MultiheadAttention modules
      - "attention_ffn": wrap attention + FFN linear layers
    """

    if placement not in ("attention_only", "attention_ffn"):
        raise ValueError(f"unknown placement: {placement}")

    cfg = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)

    def _replace_mha(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(mod, name, LoRAMultiheadAttention(child, cfg))
            else:
                _replace_mha(child)

    _replace_mha(model)

    if placement == "attention_ffn":
        # torchvision DETR uses TransformerEncoderLayer/DecoderLayer with linear1/linear2 for FFN.
        for m in model.modules():
            if hasattr(m, "linear1") and isinstance(getattr(m, "linear1"), nn.Linear):
                setattr(m, "linear1", LoRALinear(getattr(m, "linear1"), cfg))
            if hasattr(m, "linear2") and isinstance(getattr(m, "linear2"), nn.Linear):
                setattr(m, "linear2", LoRALinear(getattr(m, "linear2"), cfg))
