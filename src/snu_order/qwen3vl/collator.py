from __future__ import annotations

from typing import Any

import torch


def build_qwen3_messages(prompt: str, images: list[Any]) -> list[dict[str, Any]]:
    if len(images) != 4:
        raise ValueError(f"Qwen3 classifier expects 4 images, got {len(images)}")
    content = [{"type": "image", "image": image.convert("RGB") if hasattr(image, "convert") else image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class Qwen3VLCollator:
    def __init__(self, processor: Any | None = None, *, add_generation_prompt: bool = True) -> None:
        self.processor = processor
        self.add_generation_prompt = bool(add_generation_prompt)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch: dict[str, Any] = {
            "id": [sample["id"] for sample in samples],
            "answer": torch.tensor([sample["answer"] for sample in samples], dtype=torch.long),
            "target_perm_idx": torch.tensor([sample["target_perm_idx"] for sample in samples], dtype=torch.long),
            "position_labels": torch.tensor([sample["position_labels"] for sample in samples], dtype=torch.long),
            "pairwise_labels": torch.tensor([sample["pairwise_labels"] for sample in samples], dtype=torch.long),
        }
        if self.processor is None:
            return batch
        if len(samples) != 1:
            raise ValueError("Qwen3VLCollator currently supports batch size 1 for PIL multi-image inputs")
        sample = samples[0]
        messages = build_qwen3_messages(str(sample["prompt"]), list(sample["images"]))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        batch["inputs"] = dict(inputs)
        return batch


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "inputs":
            moved[key] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()}
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
