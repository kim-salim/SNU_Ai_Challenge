from snu_order.vlm24.candidates import build_24_candidates
from snu_order.vlm24.prompt_builder import build_prompt


def test_prompt() -> None:
    sentence = "A person opens a door and walks outside."
    candidates = build_24_candidates()
    prompt = build_prompt(sentence, candidates)
    assert sentence in prompt
    assert "Return exactly one option letter" in prompt
    assert "explain step by step" not in prompt.lower()
    for candidate in candidates:
        assert f"{candidate['label']}: {candidate['text']}" in prompt
