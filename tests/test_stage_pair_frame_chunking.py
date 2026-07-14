from types import SimpleNamespace

import pytest
import torch
from torch import nn

from snu_order.qwen3vl.frame_chunking import (
    normalize_frame_chunk_size,
    slice_frame_multimodal_inputs,
)
from snu_order.qwen3vl.modeling_stage_pair import Qwen3VLStagePairModel


def _inputs() -> dict[str, torch.Tensor]:
    grid = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4]])
    return {
        "input_ids": torch.tensor(
            [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "pixel_values": torch.arange(10 * 6, dtype=torch.float32).reshape(10, 6),
        "image_grid_thw": grid,
    }


def test_frame_multimodal_slice_preserves_rows_and_patch_order():
    inputs = _inputs()
    sliced = slice_frame_multimodal_inputs(inputs, start=1, end=3, total_frames=4)
    assert sliced["input_ids"].tolist() == [[20, 21, 22], [30, 31, 32]]
    assert sliced["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]
    assert torch.equal(sliced["pixel_values"], inputs["pixel_values"][1:6])


def test_frame_multimodal_slice_rejects_inconsistent_patch_count():
    inputs = _inputs()
    inputs["pixel_values"] = inputs["pixel_values"][:-1]
    with pytest.raises(RuntimeError, match="patch counts"):
        slice_frame_multimodal_inputs(inputs, start=0, end=1, total_frames=4)


def test_frame_chunk_size_validation_is_fail_closed():
    assert normalize_frame_chunk_size(None) is None
    assert normalize_frame_chunk_size(1) == 1
    assert normalize_frame_chunk_size(2) == 2
    assert normalize_frame_chunk_size(4) == 4
    with pytest.raises(RuntimeError, match="frame_chunk_size"):
        normalize_frame_chunk_size(3)


class _IndependentFakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sentinel = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object) -> SimpleNamespace:
        basis = torch.arange(4, device=input_ids.device, dtype=torch.float32)
        hidden = input_ids.float().unsqueeze(-1) + basis
        hidden = hidden * attention_mask.unsqueeze(-1)
        return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden)


def _model() -> Qwen3VLStagePairModel:
    torch.manual_seed(42)
    model = Qwen3VLStagePairModel(
        _IndependentFakeBackbone(),
        hidden_size=4,
        model_dim=8,
        set_layers=1,
        set_heads=2,
        set_ffn_dim=16,
        dropout=0.0,
        pooling_mode="anchor_span_mean",
    )
    return model.eval()


def test_chunk_size_4_2_1_has_full_forward_parity_and_preserves_frame_order():
    model = _model()
    inputs = _inputs()
    anchor_mask = torch.tensor([[False, True, True]] * 4)
    with torch.inference_mode():
        legacy = model(inputs=inputs, batch_size=1, anchor_mask=anchor_mask)
        chunk4 = model(inputs=inputs, batch_size=1, anchor_mask=anchor_mask, frame_chunk_size=4)
        chunk2 = model(inputs=inputs, batch_size=1, anchor_mask=anchor_mask, frame_chunk_size=2)
        chunk1 = model(inputs=inputs, batch_size=1, anchor_mask=anchor_mask, frame_chunk_size=1)
    expected_first_dimension = torch.tensor([11.5, 21.5, 31.5, 41.5])
    assert torch.equal(legacy["frame_hidden"][0, :, 0], expected_first_dimension)
    for candidate in (chunk4, chunk2, chunk1):
        for key in ("frame_hidden", "stage_logits", "pair_logits", "final_logits"):
            assert torch.equal(candidate[key], legacy[key])


def test_chunked_path_rejects_cross_sample_batching():
    model = _model()
    with pytest.raises(RuntimeError, match="batch_size=1"):
        model.extract_frame_representations(
            _inputs(),
            batch_size=2,
            anchor_mask=torch.ones(8, 3, dtype=torch.bool),
            frame_chunk_size=1,
        )
