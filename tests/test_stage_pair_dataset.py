import csv
from pathlib import Path

from PIL import Image

from snu_order.qwen3vl.dataset_single_frame import Qwen3VLSingleFrameCollator, Qwen3VLSingleFrameDataset, build_single_frame_prompt


def _make_fixture(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for idx in range(4):
        path = image_dir / f"f{idx}.png"
        Image.new("RGB", (8, 8), color=(idx * 40, 0, 0)).save(path)
        paths.append(path.relative_to(tmp_path))
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Id", "Sentence", "frame_1", "frame_2", "frame_3", "frame_4", "Answer"])
        writer.writeheader()
        writer.writerow({
            "Id": "s0",
            "Sentence": "event",
            "frame_1": str(paths[0]),
            "frame_2": str(paths[1]),
            "frame_3": str(paths[2]),
            "frame_4": str(paths[3]),
            "Answer": "[2,4,1,3]",
        })
    return csv_path


def test_single_frame_prompt_has_no_slot_labels():
    prompt = build_single_frame_prompt("event")
    assert "Frame A" not in prompt
    assert "STATE:" in prompt


def test_dataset_loads_rgb_frames_and_targets(tmp_path):
    csv_path = _make_fixture(tmp_path)
    dataset = Qwen3VLSingleFrameDataset(csv_path, tmp_path, training=False)
    sample = dataset[0]
    assert sample["id"] == "s0"
    assert sample["answer"] == [2, 4, 1, 3]
    assert sample["stage_targets"] == [1, 3, 0, 2]
    assert len(sample["images"]) == 4
    assert all(image.mode == "RGB" for image in sample["images"])


def test_validation_augmentation_off(tmp_path):
    csv_path = _make_fixture(tmp_path)
    dataset = Qwen3VLSingleFrameDataset(csv_path, tmp_path, training=False, augment_permutation=True)
    first = dataset[0]["answer"]
    dataset.set_epoch(99)
    assert dataset[0]["answer"] == first


def test_collator_without_processor(tmp_path):
    csv_path = _make_fixture(tmp_path)
    sample = Qwen3VLSingleFrameDataset(csv_path, tmp_path, training=False)[0]
    batch = Qwen3VLSingleFrameCollator(None)([sample])
    assert batch["answer"].shape == (1, 4)
    assert batch["target_perm_idx"].shape == (1,)
