from __future__ import annotations


def build_classifier_prompt(sentence: str) -> str:
    return (
        "You are given a sentence describing an event and four shuffled video frames.\n"
        "Frame A is the first supplied image.\n"
        "Frame B is the second supplied image.\n"
        "Frame C is the third supplied image.\n"
        "Frame D is the fourth supplied image.\n"
        "Determine the original temporal order of the four supplied frames.\n"
        f"Sentence: {sentence}\n"
        "ORDER:"
    )
