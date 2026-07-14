from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_json

from .calibration_stage_pair import file_sha256, permutation_table_fingerprint
from .dataset_single_frame import Qwen3VLSingleFrameDataset, build_single_frame_message
from .modeling_lora24 import load_qwen3_processor
from .stage_pair_prompt import StagePairPromptSpec, prepare_stage_pair_multimodal_inputs


LEGACY_ANCHOR = "Temporal state representation:"
LEGACY_ANCHOR_IDS = [89653, 1528, 12669, 25]
LEGACY_CANONICAL_SPAN = [294, 298]
PINNED_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(root)),
            "byte_size": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required legacy checkpoint artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _processor_for_config(cfg: dict[str, Any]) -> Any:
    processor_model_cfg = {
        "local_dir": get_by_path(cfg, "backbone.base_model_path"),
        "local_files_only": True,
        "trust_remote_code": bool(get_by_path(cfg, "backbone.trust_remote_code", True)),
        "type": get_by_path(cfg, "backbone.model_type", "qwen3_5_vl"),
        "revision": get_by_path(cfg, "backbone.revision"),
    }
    return load_qwen3_processor({"model": processor_model_cfg, "processor": cfg.get("processor", {})})


def verify_legacy_4token_checkpoint(
    checkpoint: str | Path,
    config: str | Path,
    train_metadata: str | Path,
    validation_split: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(checkpoint).resolve()
    cfg_path = Path(config).resolve()
    valid_path = Path(validation_split).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Legacy checkpoint directory not found: {root}")
    cfg = load_config(cfg_path)
    saved_cfg = load_config(root / "config.json")
    spec = StagePairPromptSpec.from_config(cfg)
    if str(get_by_path(cfg, "backbone.revision")) != PINNED_REVISION:
        raise RuntimeError("Legacy checkpoint config has the wrong pinned model revision")
    if spec.anchor_text != LEGACY_ANCHOR or spec.pooling_mode != "anchor_span_mean":
        raise RuntimeError("Legacy checkpoint does not use the exact four-token anchor pooling contract")
    if spec.enable_thinking is not False or spec.add_generation_prompt is not False or not spec.strict_template:
        raise RuntimeError("Legacy checkpoint template contract is not strict non-thinking/no-generation")
    if saved_cfg != cfg:
        raise RuntimeError("Runtime legacy config is not byte-equivalent in parsed form to checkpoint config.json")

    metrics = _load_json(root / "metrics.json")
    if int(metrics.get("sample_count", -1)) != 1430 or int(metrics.get("correct_count", -1)) != 722:
        raise RuntimeError(
            f"Legacy checkpoint native metrics are not 722/1430: "
            f"{metrics.get('correct_count')}/{metrics.get('sample_count')}"
        )
    lora_manifest = _load_json(root / "lora_target_manifest.json")
    if not isinstance(lora_manifest, list) or len(lora_manifest) != 104:
        raise RuntimeError(f"Legacy LoRA manifest must contain 104 targets, got {len(lora_manifest)}")
    groups = Counter(str(entry["layer_type"]) for entry in lora_manifest)
    if groups != Counter({"linear_attention": 72, "full_attention": 32}):
        raise RuntimeError(f"Legacy LoRA target groups mismatch: {dict(groups)}")
    for entry in lora_manifest:
        expected = (16, 32) if entry["layer_type"] == "full_attention" else (8, 16)
        if (int(entry["rank"]), int(entry["alpha"])) != expected:
            raise RuntimeError(f"Legacy LoRA rank/alpha mismatch: {entry}")
        if any(marker in str(entry["module_name"]).lower() for marker in ("visual", "vision", "lm_head")):
            raise RuntimeError(f"Legacy LoRA manifest contains a forbidden target: {entry['module_name']}")

    stored_prompt = _load_json(root / "prompt_fingerprint" / "metadata.json")
    if stored_prompt.get("anchor_text") != LEGACY_ANCHOR:
        raise RuntimeError("Stored legacy prompt fingerprint has the wrong anchor text")
    if stored_prompt.get("anchor_token_ids") != LEGACY_ANCHOR_IDS:
        raise RuntimeError("Stored legacy prompt fingerprint has unexpected anchor token IDs")
    if stored_prompt.get("anchor_spans") != [LEGACY_CANONICAL_SPAN] * 4:
        raise RuntimeError("Stored legacy prompt fingerprint has an unexpected canonical span")
    if stored_prompt.get("pooling_mode") != "anchor_span_mean":
        raise RuntimeError("Stored legacy prompt fingerprint has the wrong pooling mode")

    processor = _processor_for_config(cfg)
    dataset = Qwen3VLSingleFrameDataset(
        train_metadata,
        image_root,
        training=False,
        max_samples=1,
        prompt_spec=spec,
    )
    sample = dataset[0]
    prepared = prepare_stage_pair_multimodal_inputs(
        processor,
        [build_single_frame_message(str(sample["prompt"]), image) for image in sample["images"]],
        spec,
        model_revision=str(get_by_path(cfg, "backbone.revision")),
    )
    current_ids = prepared.inputs["input_ids"].detach().long().cpu().tolist()
    stored_tails = _load_json(root / "prompt_fingerprint" / "input_ids_tail.json")
    current_tails = [row[-len(stored):] for row, stored in zip(current_ids, stored_tails, strict=True)]
    if current_tails != stored_tails:
        raise RuntimeError("Fresh processor output tails do not match the stored legacy canonical input IDs")
    current_ids_sha256 = hashlib.sha256(
        json.dumps(current_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if current_ids_sha256 != stored_prompt.get("input_ids_sha256"):
        raise RuntimeError(
            "Fresh processor full input ID hash does not match the stored legacy canonical fingerprint"
        )
    spans = [list(value) for value in prepared.anchor_spans]
    if spans != [LEGACY_CANONICAL_SPAN] * 4:
        raise RuntimeError(f"Fresh canonical legacy anchor spans differ from [294,298): {spans}")
    mask = prepared.anchor_mask
    if mask is None or mask.shape != prepared.inputs["attention_mask"].shape:
        raise RuntimeError("Fresh legacy canonical sample has no valid anchor mask")
    for row_index, (start, end) in enumerate(spans):
        if int(mask[row_index].sum().item()) != 4 or not bool(mask[row_index, start:end].all()):
            raise RuntimeError(f"Fresh legacy anchor row {row_index} does not select exactly four tokens")
        if not bool(prepared.inputs["attention_mask"][row_index, start:end].bool().all()):
            raise RuntimeError(f"Fresh legacy anchor row {row_index} intersects padding")

    rendered = (root / "prompt_fingerprint" / "rendered_prompt.txt").read_text(encoding="utf-8")
    if prepared.rendered_prompts != [rendered] * 4:
        raise RuntimeError("Fresh rendered prompt does not match the stored legacy canonical prompt")
    processor_fingerprint = {
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": processor.tokenizer.__class__.__name__,
        "processor_config_sha256": file_sha256(root / "processor" / "processor_config.json"),
        "tokenizer_config_sha256": file_sha256(root / "processor" / "tokenizer_config.json"),
        "tokenizer_sha256": file_sha256(root / "processor" / "tokenizer.json"),
        "chat_template_sha256": file_sha256(root / "processor" / "chat_template.jinja"),
    }
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processor_path = out / "processor_fingerprint.json"
    prompt_path = out / "prompt_fingerprint.json"
    write_json(processor_path, processor_fingerprint)
    prompt_fingerprint = {
        **stored_prompt,
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "input_ids_full_sha256": current_ids_sha256,
        "fresh_processor_match": True,
        "canonical_sample_id": str(sample["id"]),
    }
    write_json(prompt_path, prompt_fingerprint)

    inventory = _directory_inventory(root)
    adapter_weight = root / "adapter" / "adapter_model.safetensors"
    if not adapter_weight.is_file():
        adapter_weight = root / "adapter" / "adapter_model.bin"
    heads = torch.load(root / "heads.pt", map_location="cpu", weights_only=False)
    if str(heads.get("pooling_mode")) != "anchor_span_mean":
        raise RuntimeError("Legacy heads.pt pooling mode mismatch")
    supplemental = {
        "format_version": "legacy-v1-supplemental-1",
        "checkpoint_directory": str(root),
        "checkpoint_directory_sha256": _canonical_sha(inventory),
        "files": inventory,
        "base_model_revision": PINNED_REVISION,
        "config_sha256": file_sha256(root / "config.json"),
        "adapter_sha256": file_sha256(adapter_weight),
        "heads_sha256": file_sha256(root / "heads.pt"),
        "prompt_fingerprint_sha256": file_sha256(prompt_path),
        "processor_fingerprint_sha256": file_sha256(processor_path),
        "permutation_mapping_sha256": file_sha256(root / "permutations.json"),
        "permutation_table_fingerprint": permutation_table_fingerprint(),
        "validation_split_sha256": file_sha256(valid_path),
        "source_code_sha256": file_sha256(Path(__file__).with_name("stage_pair_scorer.py")),
        "anchor_text": LEGACY_ANCHOR,
        "anchor_token_ids": LEGACY_ANCHOR_IDS,
        "canonical_anchor_span": LEGACY_CANONICAL_SPAN,
        "pooling_mode": "anchor_span_mean",
        "enable_thinking": False,
        "add_generation_prompt": False,
        "full_attention_target_count": 32,
        "linear_attention_target_count": 72,
        "total_lora_target_count": 104,
        "lora_trainable_parameter_count": sum(int(entry["parameter_count"]) for entry in lora_manifest),
        "vision_lora_trainable_parameter_count": 0,
        "native_raw_correct_count": 722,
        "native_raw_sample_count": 1430,
    }
    supplemental_path = out / "supplemental_checkpoint_manifest.json"
    write_json(supplemental_path, supplemental)
    write_json(out / "verification_summary.json", {"status": "PASS", **supplemental})
    return supplemental


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--validation-split", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = verify_legacy_4token_checkpoint(
        args.checkpoint,
        args.config,
        args.train_metadata,
        args.validation_split,
        args.image_root,
        args.output_dir,
    )
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
