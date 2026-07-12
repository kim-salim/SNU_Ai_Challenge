from __future__ import annotations

from pathlib import Path

from PIL import Image

from snu_order.qwen3vl.dataset import Qwen3VLFrameOrderDataset
from snu_order.qwen3vl.permutations import (
    PERMS,
    apply_frame_permutation,
    answer_to_perm_index,
    perm_index_to_answer,
    remap_answer_for_input_shuffle,
    uniform_shuffle_for_sample,
)


def test_all_24_frame_shuffle_answer_remap_preserves_meaning():
    old_answer = [2, 4, 1, 3]
    old_frames = ["F0", "F1", "F2", "F3"]
    for shuffle in PERMS:
        new_frames, new_answer = apply_frame_permutation(old_frames, old_answer, shuffle)
        assert sorted(new_answer) == [1, 2, 3, 4]
        assert new_frames == [old_frames[i] for i in shuffle]
        assert new_answer == [old_answer[i] for i in shuffle]


def test_answer_perm_index_round_trip():
    for perm in PERMS:
        answer = [0, 0, 0, 0]
        for pos, frame_idx in enumerate(perm, start=1):
            answer[frame_idx] = pos
        assert perm_index_to_answer(answer_to_perm_index(answer)) == answer


def test_same_seed_gives_same_shuffle():
    assert uniform_shuffle_for_sample(42, 7, 3) == uniform_shuffle_for_sample(42, 7, 3)


def test_validation_dataset_does_not_augment(tmp_path: Path):
    for idx in range(4):
        Image.new("RGB", (4, 4), (idx * 40, 0, 0)).save(tmp_path / f"f{idx}.png")
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "Id,Sentence,frame0,frame1,frame2,frame3,Answer\n"
        "s1,event,f0.png,f1.png,f2.png,f3.png,\"[2,4,1,3]\"\n",
        encoding="utf-8",
    )
    dataset = Qwen3VLFrameOrderDataset(
        csv_path,
        tmp_path,
        training=False,
        augment_permutation=True,
        permutation_probability=1.0,
        seed=42,
    )
    sample = dataset[0]
    assert sample["answer"] == [2, 4, 1, 3]
    assert sample["shuffle_idx"] is None


def test_remap_answer_rejects_invalid_answer():
    try:
        remap_answer_for_input_shuffle([1, 1, 2, 3], (0, 1, 2, 3))
    except ValueError:
        return
    raise AssertionError("invalid answer should raise ValueError")
