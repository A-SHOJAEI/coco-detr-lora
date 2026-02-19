import torch

from coco_detr_lora.models.build import build_model
from coco_detr_lora.models.trainability import apply_finetune_regime


def _trainable_names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_detr_lora_only_freezes_non_lora_except_heads():
    built = build_model(
        {
            "arch": "detr_r50",
            "num_classes": 3,
            "lora": {"enabled": True, "rank": 4, "alpha": 8, "dropout": 0.0, "placement": "attention_only"},
        }
    )
    model = built.model
    apply_finetune_regime(model, arch="detr_r50", regime="lora_only")

    names = _trainable_names(model)
    assert any("q_A" in n or "q_B" in n or "lora_" in n or n.endswith(("_A", "_B")) for n in names)
    assert any(n.startswith("class_embed") for n in names)
    assert any(n.startswith("bbox_embed") for n in names)

    # Backbone should be frozen in lora_only.
    assert not any(n.startswith("backbone") for n in names)


def test_detr_partial_unfreeze_enables_layer4_and_heads_only():
    built = build_model({"arch": "detr_r50", "num_classes": 3, "lora": {"enabled": False}})
    model = built.model
    apply_finetune_regime(model, arch="detr_r50", regime="partial")

    names = _trainable_names(model)
    assert any(n.startswith("backbone.body.layer4") for n in names)
    assert any(n.startswith("class_embed") for n in names)
    assert any(n.startswith("bbox_embed") for n in names)

    # Earlier backbone stages should remain frozen.
    assert not any(n.startswith("backbone.body.layer1") for n in names)
    assert not any(n.startswith("backbone.body.layer2") for n in names)
    assert not any(n.startswith("backbone.body.layer3") for n in names)


def test_full_finetune_enables_all_params():
    built = build_model({"arch": "detr_r50", "num_classes": 3, "lora": {"enabled": False}})
    model = built.model
    apply_finetune_regime(model, arch="detr_r50", regime="full")
    assert all(p.requires_grad for p in model.parameters())

