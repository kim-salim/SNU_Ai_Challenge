from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from snu_order.qwen3vl.champion_retention import (
    CACHE_FORMAT,
    ChampionRetentionLRScheduler,
    ChampionTeacherStore,
    champion_retention_loss,
    remap_teacher_logits_for_input_shuffle,
)
from snu_order.qwen3vl.modeling_stage_pair import build_stage_pair_head_from_config
from snu_order.qwen3vl.permutations import PAIRS, answer_to_perm_index, pairwise_labels_from_answer
from snu_order.qwen3vl.qwen35_27b_port import Qwen35_27BStagePairE1Model
from snu_order.qwen3vl.stage_pair_scorer import structured_permutation_logits


def _split_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cfg() -> dict:
    return {
        "architecture": {"id": "qwen35_27b_stage_pair_e1_champion_retention_v2"},
        "model": {
            "model_dim": 512,
            "set_layers": 2,
            "set_heads": 8,
            "set_ffn_dim": 2048,
            "dropout": 0.1,
            "use_set_encoder": True,
            "use_pairwise": True,
        },
        "score": {"stage_weight": 1.0, "pair_weight": 0.3},
        "pooling": {"mode": "anchor_span_mean"},
        "prompt": {
            "enable_thinking": False,
            "add_generation_prompt": False,
            "anchor_text": "STATE:",
            "anchor_prefix": "\n",
            "strict_template": True,
        },
        "backbone": {"frozen": False},
        "retention": {
            "enabled": True,
            "temperature": 2.0,
            "stage_kd_weight": 0.1,
            "pair_kd_weight": 0.05,
            "permutation_kd_weight": 0.1,
            "schedule": {
                "projector_only_fraction": 1 / 12,
                "stabilization_end_fraction": 0.25,
                "projector_only_lr": 3e-4,
                "stabilization_projector_lr": 1e-4,
                "stabilization_head_lr": 5e-5,
                "joint_lora_lr": 2e-5,
                "joint_head_lr": 3e-5,
                "joint_warmup_ratio": 0.05,
            },
        },
    }


def test_retention_architecture_uses_27b_port() -> None:
    model = build_stage_pair_head_from_config(_cfg(), hidden_size=5120, backbone=None)
    assert isinstance(model, Qwen35_27BStagePairE1Model)
    assert model.frame_projector.proj.in_features == 5120


def test_teacher_pair_remap_matches_semantic_answers() -> None:
    answer = [1, 4, 2, 3]
    pair = torch.tensor(pairwise_labels_from_answer(answer), dtype=torch.float32).mul(2).sub(1)
    stage = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    shuffle = (2, 0, 3, 1)
    remapped_stage, remapped_pair = remap_teacher_logits_for_input_shuffle(stage, pair, shuffle)
    expected_answer = [answer[index] for index in shuffle]
    expected_pair = torch.tensor(pairwise_labels_from_answer(expected_answer), dtype=torch.float32).mul(2).sub(1)
    assert torch.equal(remapped_stage, stage[list(shuffle)])
    assert torch.equal(remapped_pair, expected_pair)


def test_teacher_store_is_strict_and_shuffle_equivariant(tmp_path: Path) -> None:
    split = tmp_path / "train.csv"
    split.write_text("Id,Answer\na,1 2 3 4\n", encoding="utf-8")
    stage = torch.randn(1, 4, 4)
    pair = torch.randn(1, 6)
    target = torch.tensor([answer_to_perm_index([1, 2, 3, 4])])
    cache = tmp_path / "teacher.pt"
    torch.save(
        {
            "cache_format": CACHE_FORMAT,
            "ids": ["a"],
            "stage_logits": stage,
            "pair_logits": pair,
            "target_perm_idx": target,
            "answer": torch.tensor([[1, 2, 3, 4]]),
            "teacher_correct": torch.tensor([True]),
            "identity": {"train_split_sha256": _split_sha(split)},
        },
        cache,
    )
    store = ChampionTeacherStore(cache, expected_ids=["a"], expected_split_sha256=_split_sha(split))
    sample = store.sample("a", (3, 2, 1, 0))
    expected = structured_permutation_logits(
        sample["teacher_stage_logits"].unsqueeze(0),
        sample["teacher_pair_logits"].unsqueeze(0),
        stage_weight=1.0,
        pair_weight=0.3,
    )[0]
    assert torch.equal(sample["teacher_final_logits"], expected)


def test_retention_loss_masks_teacher_wrong() -> None:
    cfg = _cfg()
    outputs = {
        "stage_logits": torch.randn(2, 4, 4, requires_grad=True),
        "pair_logits": torch.randn(2, 6, requires_grad=True),
        "final_logits": torch.randn(2, 24, requires_grad=True),
    }
    batch = {
        "teacher_stage_logits": torch.randn(2, 4, 4),
        "teacher_pair_logits": torch.randn(2, 6),
        "teacher_final_logits": torch.randn(2, 24),
        "teacher_correct": torch.tensor([False, False]),
    }
    result = champion_retention_loss(outputs, batch, cfg)
    assert result.active_samples == 0
    assert result.loss.item() == 0.0


def test_retention_scheduler_three_phases() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter], "lr": 1.0, "group_name": "frame_projector"},
        ]
    )
    scheduler = ChampionRetentionLRScheduler(optimizer, 120, _cfg())
    assert scheduler.phase_at(0) == "projector_only"
    assert scheduler.phase_at(10) == "head_stabilization"
    assert scheduler.phase_at(30) == "joint_qlora"
    assert optimizer.param_groups[0]["lr"] == 3e-4


def test_pair_table_has_six_unordered_pairs() -> None:
    assert len(PAIRS) == 6
    assert all(left < right for left, right in PAIRS)
