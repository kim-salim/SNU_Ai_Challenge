from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from snu_order.qwen3vl.canonical_stage_pair_evaluation import (
    canonical_cpu_float32_scores,
    stable_ranking,
)
from snu_order.qwen3vl.calibration_stage_pair import save_raw_stage_pair_logits
from snu_order.qwen3vl.p0_stage_pair_parity import audit_raw_stage_pair_artifact
from snu_order.qwen3vl.runtime_repeatability_gate import audit_repeats
from snu_order.qwen3vl.stage_pair_checkpoint_v3 import _assert_port_runtime_contract


def test_stable_ranking_uses_lower_class_index_for_exact_ties() -> None:
    scores = torch.zeros((2, 24), dtype=torch.float32)
    scores[0, 9] = 1.0
    scores[0, 3] = 1.0
    ranking = stable_ranking(scores, torch.tensor([9, 23]))
    assert ranking.prediction.tolist() == [3, 0]
    assert ranking.gt_rank is not None
    assert ranking.gt_rank.tolist() == [2, 24]
    assert ranking.top1_margin.tolist() == [0.0, 0.0]


def test_p0_repeated_replay_and_training_metric_parity(tmp_path: Path) -> None:
    stage = torch.randn(5, 4, 4)
    pair = torch.randn(5, 6)
    targets = torch.tensor([0, 1, 2, 3, 4])
    scores = canonical_cpu_float32_scores(stage, pair)
    ranking = stable_ranking(scores, targets)
    assert ranking.gt_rank is not None
    raw = tmp_path / "raw.pt"
    torch.save(
        {
            "ids": [f"sample-{index}" for index in range(5)],
            "stage_logits": stage,
            "pair_logits": pair,
            "target_perm_idx": targets,
            "answer": torch.tensor([[1, 2, 3, 4]] * 5),
            "raw_fused_scores": scores,
            "raw_prediction": ranking.prediction,
            "true_class_rank": ranking.gt_rank,
        },
        raw,
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"correct_count": int(ranking.prediction.eq(targets).sum())}), encoding="utf-8")
    report = audit_raw_stage_pair_artifact(raw, training_metrics_path=metrics)
    assert report["status"] == "P0_CANONICAL_SCORER_PARITY_PASS"
    assert len({row["semantic_prediction_sha256"] for row in report["repeats"]}) == 1


def _runtime_contract_config(*, fork: bool, retention: dict, allow: bool = False) -> dict:
    return {
        "architecture": {"id": "qwen35_27b_stage_pair_e1_int4_v1"},
        "backbone": {},
        "quantization": {},
        "prompt": {},
        "pooling": {"mode": "anchor_span_mean"},
        "data": {},
        "lora": {},
        "vision_merger_lora": {"enabled": False},
        "model": {},
        "score": {},
        "retention": retention,
        "train": {"fork_after_head_warmup": fork},
        "checkpoint": {"allow_shared_warmup_retention_fork": allow},
    }


def test_shared_warmup_allows_only_explicit_component_safe_retention_diff() -> None:
    saved = _runtime_contract_config(fork=True, retention={"enabled": False})
    candidate = _runtime_contract_config(
        fork=False,
        allow=True,
        retention={"enabled": True, "mode": "component_safe_v3", "soft_kl": False},
    )
    _assert_port_runtime_contract(saved, candidate)


@pytest.mark.parametrize(
    ("fork", "allow", "retention"),
    [
        (False, True, {"enabled": True, "mode": "component_safe_v3", "soft_kl": False}),
        (True, False, {"enabled": True, "mode": "component_safe_v3", "soft_kl": False}),
        (True, True, {"enabled": True, "mode": "component_safe_v3", "soft_kl": True}),
        (True, True, {"enabled": True, "mode": "legacy_soft_kd", "soft_kl": False}),
    ],
)
def test_shared_warmup_retention_override_rejects_unsafe_cases(
    fork: bool, allow: bool, retention: dict
) -> None:
    saved = _runtime_contract_config(fork=fork, retention={"enabled": False})
    candidate = _runtime_contract_config(fork=False, allow=allow, retention=retention)
    with pytest.raises(RuntimeError, match="checkpoint/runtime contract mismatch"):
        _assert_port_runtime_contract(saved, candidate)


def test_three_fresh_process_artifacts_require_identical_predictions_and_ranks(tmp_path: Path) -> None:
    stage = torch.randn(7, 4, 4)
    pair = torch.randn(7, 6)
    target = torch.arange(7)
    paths = []
    for run in range(3):
        path = tmp_path / f"run_{run}.pt"
        save_raw_stage_pair_logits(
            path,
            ids=[f"sample-{index}" for index in range(7)],
            stage_logits=stage.clone(),
            pair_logits=pair.clone(),
            target_perm_idx=target.clone(),
            answer=torch.tensor([[1, 2, 3, 4]] * 7),
        )
        paths.append(path)
    report = audit_repeats(paths)
    assert report["status"] == "P0_FRESH_PROCESS_REPEATABILITY_PASS"
    assert all(row["prediction_mismatch"] == 0 for row in report["pairwise"])
