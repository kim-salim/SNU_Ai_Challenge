from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import transformers

from snu_order.utils.config import get_by_path
from snu_order.utils.io import write_json


ANCHOR_POOLING_MODE = "anchor_span_mean"
LEGACY_POOLING_MODE = "last_non_padding"
E1_ANCHOR_TEXT = "STATE:"
DEFAULT_ANCHOR_PREFIX = "\n"


@dataclass(frozen=True)
class StagePairPromptSpec:
    pooling_mode: str
    anchor_text: str | None
    anchor_prefix: str
    add_generation_prompt: bool
    enable_thinking: bool | None
    strict_template: bool

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> StagePairPromptSpec:
        pooling_mode = str(get_by_path(cfg, "pooling.mode", LEGACY_POOLING_MODE))
        if pooling_mode not in {LEGACY_POOLING_MODE, ANCHOR_POOLING_MODE}:
            raise RuntimeError(f"Unsupported pooling.mode: {pooling_mode}")
        anchor = get_by_path(cfg, "prompt.anchor_text", None)
        spec = cls(
            pooling_mode=pooling_mode,
            anchor_text=None if anchor is None else str(anchor),
            anchor_prefix=str(get_by_path(cfg, "prompt.anchor_prefix", DEFAULT_ANCHOR_PREFIX)),
            add_generation_prompt=bool(
                get_by_path(cfg, "prompt.add_generation_prompt", pooling_mode == LEGACY_POOLING_MODE)
            ),
            enable_thinking=get_by_path(cfg, "prompt.enable_thinking", None),
            strict_template=bool(
                get_by_path(
                    cfg,
                    "prompt.strict_template",
                    get_by_path(cfg, "prompt.require_exact_template", False),
                )
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise RuntimeError("prompt.enable_thinking must be a boolean or null")
        if self.pooling_mode != ANCHOR_POOLING_MODE:
            return
        if self.anchor_text is None or not self.anchor_text:
            raise RuntimeError("anchor_span_mean requires a nonempty prompt.anchor_text")
        if self.add_generation_prompt:
            raise RuntimeError("anchor_span_mean requires prompt.add_generation_prompt=false")
        if self.enable_thinking is not False:
            raise RuntimeError("anchor_span_mean requires prompt.enable_thinking=false")
        if not self.strict_template:
            raise RuntimeError("anchor_span_mean requires prompt.strict_template=true")


@dataclass(frozen=True)
class PreparedStagePairInputs:
    inputs: dict[str, Any]
    anchor_mask: torch.Tensor | None
    rendered_prompts: list[str]
    anchor_spans: list[tuple[int, int]]
    anchor_token_ids: list[int]
    contextual_anchor_token_ids: list[int]


def build_stage_pair_prompt(sentence: str, spec: StagePairPromptSpec) -> str:
    anchor = E1_ANCHOR_TEXT if spec.anchor_text is None else spec.anchor_text
    return (
        f"Sentence: {sentence}\n\n"
        "This image is one of four shuffled frames sampled from the described event.\n"
        "Represent the visual state and its relative progress within the described event."
        f"{spec.anchor_prefix}{anchor}"
    )


def build_stage_pair_message(prompt: str, image: Any) -> list[dict[str, Any]]:
    image = image.convert("RGB") if hasattr(image, "convert") else image
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


def _to_flat_list(value: Any, *, name: str) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError(f"Expected one tokenizer row for {name}, got {len(value)}")
        value = value[0]
    if not isinstance(value, list):
        raise RuntimeError(f"Tokenizer {name} must be list-like, got {type(value).__name__}")
    return value


def _anchor_encodings(tokenizer: Any, prefix: str, anchor_text: str) -> tuple[list[int], list[int], int, int]:
    contextual = prefix + anchor_text
    try:
        encoded = tokenizer(contextual, add_special_tokens=False, return_offsets_mapping=True)
    except Exception as exc:
        raise RuntimeError("Tokenizer must support offset mapping for exact STATE anchor alignment") from exc
    contextual_ids = [int(value) for value in _to_flat_list(encoded["input_ids"], name="input_ids")]
    offsets = _to_flat_list(encoded["offset_mapping"], name="offset_mapping")
    if len(offsets) != len(contextual_ids):
        raise RuntimeError("Tokenizer returned inconsistent anchor offsets")
    anchor_start_char = len(prefix)
    selected = [
        index
        for index, offset in enumerate(offsets)
        if int(offset[1]) > anchor_start_char and int(offset[0]) < len(contextual)
    ]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise RuntimeError("STATE anchor does not align to a contiguous token span")
    start, end = selected[0], selected[-1] + 1
    return contextual_ids, contextual_ids[start:end], start, end


def _subsequence_matches(sequence: list[int], candidate: list[int]) -> list[int]:
    if not candidate or len(candidate) > len(sequence):
        return []
    width = len(candidate)
    return [index for index in range(len(sequence) - width + 1) if sequence[index : index + width] == candidate]


def _forbidden_ids(tokenizer: Any) -> set[int]:
    forbidden = {int(value) for value in (getattr(tokenizer, "all_special_ids", None) or [])}
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        forbidden.add(int(pad_id))
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        unknown = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"):
            token_id = converter(token)
            if token_id is not None and (unknown is None or int(token_id) != int(unknown)):
                forbidden.add(int(token_id))
    return forbidden


def _anchor_error(
    tokenizer: Any,
    row_index: int,
    sequence: list[int],
    anchor_ids: list[int],
    contextual_ids: list[int],
) -> RuntimeError:
    tail = sequence[-64:]
    try:
        decoded_tail = tokenizer.decode(tail, skip_special_tokens=False)
    except Exception:
        decoded_tail = "<decode failed>"
    detail = {
        "row": row_index,
        "decoded_tail": decoded_tail,
        "input_ids_tail": tail,
        "anchor_candidate_ids": anchor_ids,
        "contextual_anchor_candidate_ids": contextual_ids,
    }
    return RuntimeError(f"STATE anchor span was not found in processor input_ids: {json.dumps(detail, ensure_ascii=False)}")


def locate_anchor_spans(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    rendered_prompts: Sequence[str],
    anchor_text: str,
    *,
    anchor_prefix: str = DEFAULT_ANCHOR_PREFIX,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[int], list[int]]:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise RuntimeError(
            f"input_ids and attention_mask must share [B,L], got {tuple(input_ids.shape)} and "
            f"{tuple(attention_mask.shape)}"
        )
    if len(rendered_prompts) != input_ids.shape[0]:
        raise RuntimeError("Rendered prompt count does not match processor batch size")
    contextual_ids, anchor_ids, relative_start, relative_end = _anchor_encodings(
        tokenizer, anchor_prefix, anchor_text
    )
    forbidden = _forbidden_ids(tokenizer)
    if set(anchor_ids) & forbidden:
        raise RuntimeError(f"Anchor candidate IDs overlap special/image/pad IDs: {sorted(set(anchor_ids) & forbidden)}")
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    spans: list[tuple[int, int]] = []
    contextual_text = anchor_prefix + anchor_text
    for row_index, rendered in enumerate(rendered_prompts):
        if contextual_text not in rendered and anchor_text not in rendered:
            raise _anchor_error(tokenizer, row_index, input_ids[row_index].tolist(), anchor_ids, contextual_ids)
        sequence = [int(value) for value in input_ids[row_index].detach().cpu().tolist()]
        contextual_matches = _subsequence_matches(sequence, contextual_ids)
        if contextual_matches:
            context_start = contextual_matches[-1]
            start = context_start + relative_start
            end = context_start + relative_end
        else:
            anchor_matches = _subsequence_matches(sequence, anchor_ids)
            if not anchor_matches:
                raise _anchor_error(tokenizer, row_index, sequence, anchor_ids, contextual_ids)
            start = anchor_matches[-1]
            end = start + len(anchor_ids)
        if not bool(attention_mask[row_index, start:end].bool().all()):
            raise RuntimeError(f"STATE anchor row {row_index} intersects padding at span {(start, end)}")
        selected = set(sequence[start:end])
        if selected & forbidden:
            raise RuntimeError(
                f"STATE anchor row {row_index} includes special/image/pad IDs: {sorted(selected & forbidden)}"
            )
        mask[row_index, start:end] = True
        spans.append((start, end))
    return mask, spans, anchor_ids, contextual_ids


class AnchorSpanMeanPooler(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise RuntimeError(f"hidden_states must have [B,L,H], got {tuple(hidden_states.shape)}")
        if anchor_mask.shape != hidden_states.shape[:2]:
            raise RuntimeError(
                f"anchor_mask must have {tuple(hidden_states.shape[:2])}, got {tuple(anchor_mask.shape)}"
            )
        if not bool(anchor_mask.bool().any(dim=1).all()):
            raise RuntimeError("anchor_mask contains an all-zero row")
        mask = anchor_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)


def _strict_template_error(processor: Any, revision: str | None, exc: Exception | None = None) -> RuntimeError:
    message = (
        "Installed processor does not support strict enable_thinking=False; "
        f"transformers={transformers.__version__}, processor={processor.__class__.__name__}, "
        f"model_revision={revision or 'unspecified'}"
    )
    error = RuntimeError(message)
    if exc is not None:
        error.__cause__ = exc
    return error


def apply_chat_template_strict(
    processor: Any,
    conversations: list[Any],
    spec: StagePairPromptSpec,
    *,
    model_revision: str | None,
) -> list[str]:
    template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    )
    if spec.strict_template and spec.enable_thinking is False and (
        template is None or "enable_thinking" not in str(template)
    ):
        raise _strict_template_error(processor, model_revision)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": spec.add_generation_prompt,
    }
    if spec.enable_thinking is not None:
        kwargs["enable_thinking"] = spec.enable_thinking
    try:
        rendered = processor.apply_chat_template(conversations, **kwargs)
    except TypeError as exc:
        if spec.enable_thinking is False:
            raise _strict_template_error(processor, model_revision, exc)
        raise RuntimeError("Processor rejected configured chat-template arguments") from exc
    if isinstance(rendered, str):
        return [rendered]
    values = [str(value) for value in rendered]
    if len(values) != len(conversations):
        raise RuntimeError("Processor returned a different number of rendered prompts than conversations")
    return values


def prepare_stage_pair_multimodal_inputs(
    processor: Any,
    conversations: list[Any],
    spec: StagePairPromptSpec,
    *,
    model_revision: str | None,
) -> PreparedStagePairInputs:
    if not conversations:
        raise RuntimeError("Cannot prepare an empty stage-pair conversation batch")
    rendered = apply_chat_template_strict(
        processor, conversations, spec, model_revision=model_revision
    )
    images: list[Any] = []
    for row_index, conversation in enumerate(conversations):
        row_images = [
            part.get("image")
            for message in conversation
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        if len(row_images) != 1 or row_images[0] is None:
            raise RuntimeError(f"Conversation row {row_index} must contain exactly one materialized image")
        images.append(row_images[0])
    try:
        encoded = processor(
            text=rendered,
            images=images,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to tokenize the exact rendered multimodal prompt") from exc
    inputs = dict(encoded)
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise RuntimeError("Processor output is missing input_ids or attention_mask")
    if spec.pooling_mode == ANCHOR_POOLING_MODE:
        if spec.anchor_text is None:
            raise RuntimeError("Anchor pooling is missing prompt.anchor_text")
        anchor_mask, spans, anchor_ids, contextual_ids = locate_anchor_spans(
            inputs["input_ids"],
            inputs["attention_mask"],
            processor.tokenizer,
            rendered,
            spec.anchor_text,
            anchor_prefix=spec.anchor_prefix,
        )
    else:
        anchor_mask, spans, anchor_ids, contextual_ids = None, [], [], []
    return PreparedStagePairInputs(inputs, anchor_mask, rendered, spans, anchor_ids, contextual_ids)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _tokenizer_config_v1(tokenizer: Any) -> dict[str, Any]:
    return {
        "class": tokenizer.__class__.__name__,
        "init_kwargs": getattr(tokenizer, "init_kwargs", {}),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }


def _tokenizer_config_v2(tokenizer: Any) -> dict[str, Any]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else None
    added_vocab = get_added_vocab() if callable(get_added_vocab) else None
    return {
        "class": tokenizer.__class__.__name__,
        "vocab": vocab,
        "added_vocab": added_vocab,
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "all_special_tokens": getattr(tokenizer, "all_special_tokens", None),
        "all_special_ids": getattr(tokenizer, "all_special_ids", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "clean_up_tokenization_spaces": getattr(tokenizer, "clean_up_tokenization_spaces", None),
    }


def build_prompt_fingerprint(
    cfg: dict[str, Any],
    processor: Any,
    *,
    format_version: int = 2,
) -> dict[str, Any]:
    from PIL import Image

    spec = StagePairPromptSpec.from_config(cfg)
    synthetic_sentence = "A person performs an action."
    image = Image.new("RGB", (56, 56), color=(0, 0, 0))
    prompt = build_stage_pair_prompt(synthetic_sentence, spec)
    prepared = prepare_stage_pair_multimodal_inputs(
        processor,
        [build_stage_pair_message(prompt, image)],
        spec,
        model_revision=str(get_by_path(cfg, "backbone.revision", "")) or None,
    )
    tokenizer = processor.tokenizer
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        raise RuntimeError("Processor/tokenizer has no chat template for fingerprinting")
    if int(format_version) == 1:
        tokenizer_config = _tokenizer_config_v1(tokenizer)
    elif int(format_version) == 2:
        tokenizer_config = _tokenizer_config_v2(tokenizer)
    else:
        raise RuntimeError(f"Unsupported prompt fingerprint format: {format_version}")
    input_ids = [int(value) for value in prepared.inputs["input_ids"][0].detach().cpu().tolist()]
    tail = input_ids[-64:]
    grid = prepared.inputs.get("image_grid_thw")
    return {
        "format_version": int(format_version),
        "base_model_path": str(get_by_path(cfg, "backbone.base_model_path")),
        "model_revision": str(get_by_path(cfg, "backbone.revision")),
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "transformers_version": transformers.__version__,
        "peft_version": _package_version("peft"),
        "torch_version": torch.__version__,
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "chat_template_sha256": _sha256_text(str(chat_template)),
        "tokenizer_config_sha256": _sha256_text(_canonical_json(tokenizer_config)),
        "rendered_prompt": prepared.rendered_prompts[0],
        "rendered_prompt_sha256": _sha256_text(prepared.rendered_prompts[0]),
        "input_ids_sha256": _sha256_text(_canonical_json(input_ids)),
        "input_ids_length": len(input_ids),
        "input_ids_tail_64": tail,
        "decoded_tail_64": tokenizer.decode(tail, skip_special_tokens=False),
        "anchor_text": spec.anchor_text,
        "anchor_token_ids": prepared.anchor_token_ids,
        "anchor_span": list(prepared.anchor_spans[0]) if prepared.anchor_spans else None,
        "enable_thinking": spec.enable_thinking,
        "add_generation_prompt": spec.add_generation_prompt,
        "pooling_mode": spec.pooling_mode,
        "image_grid_thw": None if grid is None else grid.detach().cpu().tolist(),
    }


FINGERPRINT_COMPARISON_FIELDS = (
    "format_version",
    "base_model_path",
    "model_revision",
    "processor_class",
    "tokenizer_class",
    "chat_template_sha256",
    "tokenizer_config_sha256",
    "rendered_prompt_sha256",
    "input_ids_sha256",
    "anchor_text",
    "anchor_token_ids",
    "anchor_span",
    "enable_thinking",
    "add_generation_prompt",
    "pooling_mode",
)


def checkpoint_processor_path(checkpoint: str | Path) -> Path | None:
    root = Path(checkpoint)
    if not (root / "checkpoint_manifest.json").is_file():
        return None
    fingerprint_path = root / "prompt_fingerprint.json"
    if not fingerprint_path.is_file():
        raise RuntimeError(f"Checkpoint prompt fingerprint is missing: {fingerprint_path}")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    version = int(fingerprint.get("format_version", 1))
    if version == 1:
        # v1 included absolute tokenizer file paths and must be recreated from the pinned base cache.
        return None
    if version == 2:
        processor_path = root / "processor"
        if not processor_path.is_dir():
            raise RuntimeError(f"Checkpoint processor directory is missing: {processor_path}")
        return processor_path
    raise RuntimeError(f"Unsupported checkpoint prompt fingerprint format: {version}")


def assert_prompt_fingerprint_match(saved: dict[str, Any], current: dict[str, Any]) -> None:
    mismatches = {
        field: {"checkpoint": saved.get(field), "runtime": current.get(field)}
        for field in FINGERPRINT_COMPARISON_FIELDS
        if saved.get(field) != current.get(field)
    }
    if mismatches:
        raise RuntimeError(f"Prompt fingerprint mismatch: {json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}")


def write_prompt_fingerprint(path: str | Path, fingerprint: dict[str, Any]) -> None:
    write_json(Path(path), fingerprint)
