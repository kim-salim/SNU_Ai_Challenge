import json

import pytest
import torch

from snu_order.qwen3vl.calibration_stage_pair import load_raw_stage_pair_logits
from snu_order.qwen3vl.import_legacy_stage_pair_raw import import_legacy_raw_scores
from snu_order.qwen3vl.permutations import perm_index_to_answer
from snu_order.qwen3vl.stage_pair_scorer import structured_permutation_logits


def _write_source(path, *, corrupt=False):
    torch.manual_seed(4)
    stage = torch.randn(3, 4, 4)
    pair = torch.randn(3, 6)
    fused = structured_permutation_logits(stage, pair, stage_weight=1.0, pair_weight=0.3)
    targets = torch.tensor([0, 5, 23])
    with path.open("w", encoding="utf-8") as handle:
        for index in range(3):
            logits = fused[index].tolist()
            if corrupt and index == 1:
                logits[0] += 0.1
            handle.write(
                json.dumps(
                    {
                        "Id": f"sample-{index}",
                        "logits": logits,
                        "true_perm_idx": int(targets[index]),
                        "pred_perm_idx": int(fused[index].argmax()),
                        "stage_logits": stage[index].tolist(),
                        "pair_logits": pair[index].tolist(),
                    }
                )
                + "\n"
            )


def test_import_legacy_raw_scores_roundtrip(tmp_path):
    source = tmp_path / "raw.jsonl"
    destination = tmp_path / "raw.pt"
    _write_source(source)
    report = import_legacy_raw_scores(source, destination, expected_count=3)
    payload = load_raw_stage_pair_logits(destination)
    assert report["status"] == "PASS"
    assert report["max_abs_diff"] < 1e-5
    assert payload["ids"] == ["sample-0", "sample-1", "sample-2"]
    assert payload["answer"].tolist() == [perm_index_to_answer(value) for value in (0, 5, 23)]


def test_import_legacy_raw_scores_rejects_scorer_mismatch(tmp_path):
    source = tmp_path / "raw.jsonl"
    _write_source(source, corrupt=True)
    with pytest.raises(RuntimeError, match="do not reconstruct"):
        import_legacy_raw_scores(source, tmp_path / "raw.pt", expected_count=3)
