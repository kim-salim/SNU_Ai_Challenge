from __future__ import annotations

from collections.abc import Sequence
from typing import Any


DEFAULT_FRAME_LABELS = ["F1", "F2", "F3", "F4"]


def build_prompt(
    sentence: str,
    candidates: Sequence[dict[str, Any]],
    frame_labels: Sequence[str] = DEFAULT_FRAME_LABELS,
    use_cot: bool = False,
) -> str:
    if len(candidates) != 24:
        raise ValueError(f"prompt requires 24 candidates, got {len(candidates)}")
    if len(frame_labels) != 4:
        raise ValueError(f"frame_labels must have length 4, got {len(frame_labels)}")
    options = "\n".join(f"{candidate['label']}: {candidate['text']}" for candidate in candidates)
    reasoning = "Think briefly, but return only the final option letter.\n" if use_cot else ""
    return (
        "You are solving a video frame ordering task.\n\n"
        "The sentence describes the original chronological event.\n"
        f"The four frames are shuffled and labeled {', '.join(frame_labels)} in the same order as the provided images.\n\n"
        "Choose the option that lists the frames from earliest to latest.\n\n"
        "Return exactly one option letter from A to X.\n"
        "Do not explain.\n"
        f"{reasoning}\n"
        "Sentence:\n"
        f"{sentence}\n\n"
        "Options:\n"
        f"{options}\n\n"
        "Answer:"
    )
