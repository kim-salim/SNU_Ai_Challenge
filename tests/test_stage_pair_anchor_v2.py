from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from snu_order.qwen3vl.stage_pair_prompt import (
    ANCHOR_POOLING_MODE,
    AnchorSpanMeanPooler,
    StagePairPromptSpec,
    apply_chat_template_strict,
    assert_prompt_fingerprint_match,
    build_prompt_fingerprint,
    locate_anchor_spans,
)


class FakeTokenizer:
    all_special_ids = [0, 99]
    pad_token_id = 0
    unk_token_id = -1
    padding_side = "right"
    vocab_size = 1000
    init_kwargs = {"fixture": True}
    special_tokens_map = {"pad_token": "<pad>"}

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        if text == "\nSTATE:":
            ids = [9, 10, 11]
            offsets = [(0, 1), (1, 6), (6, 7)]
        else:
            ids = [10, 11]
            offsets = [(0, 5), (5, 6)]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def convert_tokens_to_ids(self, token):
        return 99 if token.startswith("<|") else -1

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(value) for value in ids)


def _locate(input_ids, attention_mask):
    return locate_anchor_spans(
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        FakeTokenizer(),
        ["prompt\nSTATE:<|im_end|>" for _ in input_ids],
        "STATE:",
    )


def test_anchor_locator_handles_right_and_left_padding():
    mask, spans, anchor_ids, _ = _locate(
        [[5, 9, 10, 11, 99, 0], [0, 0, 5, 9, 10, 11]],
        [[1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]],
    )
    assert spans == [(2, 4), (4, 6)]
    assert anchor_ids == [10, 11]
    assert mask[0].nonzero().flatten().tolist() == [2, 3]
    assert mask[1].nonzero().flatten().tolist() == [4, 5]


def test_anchor_locator_uses_last_occurrence_and_excludes_message_suffix():
    mask, spans, _, _ = locate_anchor_spans(
        torch.tensor([[9, 10, 11, 7, 9, 10, 11, 99]]),
        torch.ones((1, 8), dtype=torch.long),
        FakeTokenizer(),
        ["STATE: repeated\nSTATE:<|im_end|>"],
        "STATE:",
    )
    assert spans == [(5, 7)]
    assert mask[0, 7].item() is False


def test_anchor_locator_missing_fails_with_diagnostics():
    with pytest.raises(RuntimeError, match="input_ids_tail"):
        _locate([[5, 6, 7, 0]], [[1, 1, 1, 0]])


def test_anchor_span_mean_matches_manual_mean_and_rejects_empty_rows():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.bool)
    pooled = AnchorSpanMeanPooler()(hidden, mask)
    assert torch.allclose(pooled[0], (hidden[0, 1] + hidden[0, 2]) / 2)
    assert torch.allclose(pooled[1], (hidden[1, 0] + hidden[1, 3]) / 2)
    with pytest.raises(RuntimeError, match="all-zero"):
        AnchorSpanMeanPooler()(hidden, torch.zeros_like(mask))


class FakeProcessor:
    chat_template = "{% if enable_thinking %}think{% endif %}"

    def __init__(self, *, fail_template=False):
        self.tokenizer = FakeTokenizer()
        self.fail_template = fail_template
        self.last_kwargs = None

    def apply_chat_template(self, conversations, **kwargs):
        self.last_kwargs = kwargs
        if self.fail_template:
            raise TypeError("unsupported")
        return ["user image\nSTATE:<|im_end|>" for _ in conversations]

    def __call__(self, **kwargs):
        rows = len(kwargs["text"])
        return {
            "input_ids": torch.tensor([[5, 9, 10, 11, 99]] * rows),
            "attention_mask": torch.ones((rows, 5), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * rows),
        }


def _cfg():
    return {
        "backbone": {"base_model_path": "Qwen/Qwen3.5-9B", "revision": "fixed"},
        "prompt": {
            "enable_thinking": False,
            "add_generation_prompt": False,
            "anchor_text": "STATE:",
            "anchor_prefix": "\n",
            "strict_template": True,
        },
        "pooling": {"mode": ANCHOR_POOLING_MODE},
    }


def test_strict_template_passes_non_thinking_and_no_generation_prompt():
    processor = FakeProcessor()
    spec = StagePairPromptSpec.from_config(_cfg())
    apply_chat_template_strict(processor, [[{"role": "user", "content": []}]], spec, model_revision="fixed")
    assert processor.last_kwargs["enable_thinking"] is False
    assert processor.last_kwargs["add_generation_prompt"] is False


def test_enable_thinking_type_error_becomes_runtime_error_without_fallback():
    processor = FakeProcessor(fail_template=True)
    spec = StagePairPromptSpec.from_config(_cfg())
    with pytest.raises(RuntimeError, match="transformers=.*processor=FakeProcessor.*model_revision=fixed"):
        apply_chat_template_strict(processor, [[{"role": "user", "content": []}]], spec, model_revision="fixed")
    assert processor.last_kwargs["enable_thinking"] is False


def test_prompt_fingerprint_is_deterministic_and_mismatch_is_detected():
    processor = FakeProcessor()
    first = build_prompt_fingerprint(_cfg(), processor)
    second = build_prompt_fingerprint(_cfg(), processor)
    assert first == second
    assert first["anchor_token_ids"] == [10, 11]
    changed = dict(second)
    changed["input_ids_sha256"] = "different"
    with pytest.raises(RuntimeError, match="Prompt fingerprint mismatch"):
        assert_prompt_fingerprint_match(first, changed)
