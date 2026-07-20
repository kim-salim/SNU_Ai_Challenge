from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_json
from snu_order.utils.seed import seed_everything

from .dataset_single_frame import (
    CachedStagePairDataset,
    Qwen3VLSingleFrameCollator,
    Qwen3VLSingleFrameDataset,
    collate_cached_stage_pair,
    move_stage_pair_batch_to_device,
)
from .lockbox import guard_not_lockbox_path
from .gradient_health import GradientHealthMonitor
from .metrics_stage_pair import compare_with_baseline, compute_stage_pair_metrics, write_stage_pair_artifacts
from .modeling_stage_pair import (
    build_stage_pair_head_from_config,
    build_stage_pair_model_from_config,
    dump_version_report,
    load_stage_pair_checkpoint,
    save_stage_pair_checkpoint,
    write_stage_pair_trainable_report,
)
from .stage_pair_scorer import StagePairStructuredLoss, remap_logits_to_canonical
from .stage_pair_prompt import StagePairPromptSpec, build_prompt_fingerprint, write_prompt_fingerprint
from .train_lora24 import (
    DistributedState,
    _amp_dtype_and_scaler_enabled,
    _barrier,
    _broadcast_stop_flag,
    _decay_groups,
    _git_report,
    _init_distributed,
    _model_device,
    _scheduler,
    _seed_worker,
)
from .checkpoint import unwrap_model
from .calibration_stage_pair import save_raw_stage_pair_logits
from .lora_targets import validate_gradient_contract
from .champion_retention import (
    ChampionRetentionLRScheduler,
    ChampionTeacherStore,
    champion_retention_loss,
    retention_contract,
)
from .qwen35_27b_port import is_27b_port_config
from .canonical_stage_pair_evaluation import canonical_cpu_float32_scores
from .retention_v3 import (
    ComponentReferenceStore,
    apply_preregistered_component_coverage,
    component_safe_contract,
    component_safe_retention_loss,
)


MODES = ("frozen_stage", "frozen_stage_set", "frozen_stage_pair", "qlora_stage_pair")


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = deepcopy(cfg)
    if getattr(args, "run_id", None):
        out.setdefault("experiment", {})["run_id"] = str(args.run_id)
    if args.output_dir:
        out.setdefault("output", {})["dir"] = args.output_dir
    if args.seed is not None:
        out.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.source:
        out.setdefault("backbone", {})["source"] = args.source
    if args.max_samples is not None:
        out.setdefault("train", {})["max_samples"] = int(args.max_samples)
    if getattr(args, "max_valid_samples", None) is not None:
        out.setdefault("eval", {})["max_samples"] = int(args.max_valid_samples)
    if getattr(args, "epochs", None) is not None:
        out.setdefault("train", {})["epochs"] = int(args.epochs)
    if getattr(args, "init_head_from", None):
        out.setdefault("train", {})["init_head_from"] = str(args.init_head_from)
    if getattr(args, "teacher_cache", None):
        out.setdefault("retention", {})["teacher_cache"] = str(args.teacher_cache)
    if getattr(args, "train_split", None):
        out.setdefault("data", {})["train_split"] = str(args.train_split)
    if getattr(args, "valid_split", None):
        out.setdefault("data", {})["valid_split"] = str(args.valid_split)
    if getattr(args, "image_root", None):
        out.setdefault("data", {})["image_root"] = str(args.image_root)
    if getattr(args, "base_model_path", None):
        out.setdefault("backbone", {})["base_model_path"] = str(args.base_model_path)
    if getattr(args, "base_model_revision", None):
        out.setdefault("backbone", {})["revision"] = str(args.base_model_revision)
    if getattr(args, "fork_after_head_warmup", False):
        out.setdefault("train", {})["fork_after_head_warmup"] = True
    if getattr(args, "resume_training_state", False):
        out.setdefault("train", {})["resume_training_state"] = True
    return out


def _split_hash(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_rank_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rank_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _branch_rng_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_name(f"{path.name}_branch_rng")


def _first_forward_sha256(outputs: dict[str, torch.Tensor], sample_ids: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(json.dumps([str(value) for value in sample_ids], separators=(",", ":")).encode("utf-8"))
    for key in ("stage_logits", "pair_logits", "final_logits"):
        value = outputs[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(key.encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _run_id(cfg: dict[str, Any], mode: str) -> str:
    explicit = get_by_path(cfg, "experiment.run_id", None)
    if explicit:
        return str(explicit)
    source = str(get_by_path(cfg, "backbone.source", "base"))
    if mode == "qlora_stage_pair":
        return "qlora_stage_pair"
    suffix = mode.replace("frozen_", "frozen_")
    return f"{source}_{suffix}"


def _configure_mode(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("model", {})
    if mode == "frozen_stage":
        out["model"]["use_set_encoder"] = False
        out["model"]["use_pairwise"] = False
        out.setdefault("loss", {})["pair_weight"] = 0.0
        out.setdefault("score", {})["pair_weight"] = 0.0
    elif mode == "frozen_stage_set":
        out["model"]["use_set_encoder"] = True
        out["model"]["use_pairwise"] = False
        out.setdefault("loss", {})["pair_weight"] = 0.0
        out.setdefault("score", {})["pair_weight"] = 0.0
    elif mode in {"frozen_stage_pair", "qlora_stage_pair"}:
        out["model"]["use_set_encoder"] = True
        out["model"]["use_pairwise"] = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode.startswith("frozen"):
        out.setdefault("backbone", {})["frozen"] = True
        out.setdefault("loss", {})["consistency_weight"] = 0.0
    else:
        out.setdefault("backbone", {})["frozen"] = False
    return out


def _cache_path(cfg: dict[str, Any], split: str) -> Path:
    source = str(get_by_path(cfg, "backbone.source", "base"))
    default_name = "train_ab.pt" if split == "train" else "valid_a.pt"
    name = str(get_by_path(cfg, f"cache.{split}_filename", default_name))
    return Path(str(get_by_path(cfg, "cache.dir", "outputs/features/qwen3vl_stage_pair"))) / source / name


def _make_dataloaders(
    cfg: dict[str, Any],
    mode: str,
    ddp: DistributedState,
    processor: Any | None,
) -> tuple[DataLoader, DataLoader | None, int]:
    max_samples = int(get_by_path(cfg, "train.max_samples", -1))
    max_valid_samples = int(get_by_path(cfg, "eval.max_samples", -1))
    if mode.startswith("frozen"):
        is_27b = is_27b_port_config(cfg)
        expected_hidden = int(get_by_path(cfg, "cache.hidden_size", 5120)) if is_27b else None
        train_dataset = CachedStagePairDataset(
            _cache_path(cfg, "train"),
            max_samples=max_samples if max_samples >= 0 else None,
            preserve_dtype=is_27b,
            expected_hidden_size=expected_hidden,
        )
        valid_dataset = CachedStagePairDataset(
            _cache_path(cfg, "valid"),
            max_samples=max_valid_samples if max_valid_samples >= 0 else None,
            preserve_dtype=is_27b,
            expected_hidden_size=expected_hidden,
        )
        train_sampler = DistributedSampler(train_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True) if ddp.enabled else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(get_by_path(cfg, "train.batch_size", 64)),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_cached_stage_pair,
            num_workers=int(get_by_path(cfg, "train.num_workers", 0)),
        )
        valid_loader = DataLoader(valid_dataset, batch_size=int(get_by_path(cfg, "eval.batch_size", 128)), shuffle=False, collate_fn=collate_cached_stage_pair) if ddp.is_main else None
        return train_loader, valid_loader, int(train_dataset.frame_hidden.shape[-1])

    train_split = str(get_by_path(cfg, "data.train_split"))
    valid_split = str(get_by_path(cfg, "data.valid_split"))
    guard_not_lockbox_path(train_split, purpose="stage-pair training")
    guard_not_lockbox_path(valid_split, purpose="stage-pair early stopping")
    prompt_spec = StagePairPromptSpec.from_config(cfg)
    train_dataset = Qwen3VLSingleFrameDataset(
        train_split,
        str(get_by_path(cfg, "data.image_root", "data/raw")),
        training=True,
        augment_permutation=bool(get_by_path(cfg, "data.train_permutation_augmentation", True)),
        permutation_probability=float(get_by_path(cfg, "data.permutation_probability", 1.0)),
        seed=int(get_by_path(cfg, "experiment.seed", 42)),
        max_samples=max_samples if max_samples >= 0 else None,
        prompt_spec=prompt_spec,
    )
    if bool(get_by_path(cfg, "retention.enabled", False)):
        retention_mode = str(get_by_path(cfg, "retention.mode", "legacy_soft_kd"))
        teacher_path = Path(str(get_by_path(cfg, "retention.teacher_cache", "")))
        if not teacher_path.is_file():
            raise FileNotFoundError(f"Champion teacher cache is missing: {teacher_path}")
        expected_ids = [str(row[train_dataset.id_col]) for row in train_dataset.rows]
        if retention_mode == "component_safe_v3":
            if bool(get_by_path(cfg, "retention.soft_kl", False)):
                raise RuntimeError("Retention v3 forbids soft KL")
            reference_path = Path(str(get_by_path(cfg, "retention.reference_cache", "")))
            if not reference_path.is_file():
                raise FileNotFoundError(f"v1 reference cache is missing: {reference_path}")
            train_dataset.teacher_store = ComponentReferenceStore(
                teacher_path,
                prefix="teacher",
                expected_ids=expected_ids,
                expected_split_sha256=_split_hash(train_split),
                expected_heads_sha256=str(get_by_path(cfg, "retention.teacher_heads_sha256", "")) or None,
            )
            train_dataset.reference_store = ComponentReferenceStore(
                reference_path,
                prefix="reference",
                expected_ids=expected_ids,
                expected_split_sha256=_split_hash(train_split),
                expected_heads_sha256=str(get_by_path(cfg, "retention.reference_heads_sha256", "")) or None,
            )
        else:
            if str(get_by_path(cfg, "retention.teacher_mask_policy", "")) != "teacher_correct_only":
                raise RuntimeError("Legacy Champion retention requires teacher_correct_only masking")
            train_dataset.teacher_store = ChampionTeacherStore(
                teacher_path,
                expected_ids=expected_ids,
                expected_split_sha256=_split_hash(train_split),
            )
    valid_dataset = Qwen3VLSingleFrameDataset(
        valid_split,
        str(get_by_path(cfg, "data.image_root", "data/raw")),
        training=False,
        max_samples=max_valid_samples if max_valid_samples >= 0 else None,
        prompt_spec=prompt_spec,
    )
    model_revision = str(get_by_path(cfg, "backbone.revision", "")) or None
    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True) if ddp.enabled else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(get_by_path(cfg, "train.micro_batch_size_per_gpu", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=Qwen3VLSingleFrameCollator(
            processor,
            prompt_spec=prompt_spec,
            model_revision=model_revision,
        ),
        num_workers=int(get_by_path(cfg, "train.num_workers", 0)),
        worker_init_fn=_seed_worker,
    )
    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=Qwen3VLSingleFrameCollator(
                processor,
                prompt_spec=prompt_spec,
                model_revision=model_revision,
            ),
            num_workers=0,
        )
        if ddp.is_main
        else None
    )
    return train_loader, valid_loader, int(get_by_path(cfg, "backbone.hidden_size", 4096))


def _build_model(cfg: dict[str, Any], mode: str) -> tuple[torch.nn.Module, Any | None]:
    if mode.startswith("frozen"):
        train_cache = torch.load(_cache_path(cfg, "train"), map_location="cpu")
        if is_27b_port_config(cfg):
            from .qwen35_27b_port import assert_27b_cache_compatible

            assert_27b_cache_compatible(train_cache)
        hidden_size = int(train_cache["frame_hidden"].shape[-1])
        return build_stage_pair_head_from_config(cfg, hidden_size=hidden_size, backbone=None), None
    return build_stage_pair_model_from_config(cfg, live_backbone=True)


def _force_ddp_single_device_map(cfg: dict[str, Any], ddp: DistributedState, mode: str) -> None:
    if not ddp.enabled or mode.startswith("frozen"):
        return
    cfg.setdefault("backbone", {})["device_map"] = {"": ddp.local_rank}


def _move_stage_pair_modules(model: torch.nn.Module, device: torch.device) -> None:
    raw = unwrap_model(model)
    is_27b = str(getattr(raw, "__class__", type(raw)).__name__) == "Qwen35_27BStagePairE1Model"
    dtype = torch.bfloat16 if is_27b else None
    if getattr(raw, "frame_projector", None) is not None:
        raw.frame_projector.to(device=device, dtype=dtype)
    raw.set_encoder.to(device=device, dtype=dtype)
    raw.stage_head.to(device=device, dtype=dtype)
    if raw.pair_head is not None:
        raw.pair_head.to(device=device, dtype=dtype)


def _build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    raw = unwrap_model(model)
    weight_decay = float(get_by_path(cfg, "train.weight_decay", 0.01))
    groups: list[dict[str, Any]] = []
    backbone = getattr(raw, "backbone", None)
    def add_group(name: str, named_params: list[tuple[str, torch.nn.Parameter]], lr: float) -> None:
        tagged = _decay_groups(
            named_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        for group in tagged:
            group["group_name"] = name
        groups.extend(tagged)

    if backbone is not None:
        add_group(
            "backbone_lora",
            [(f"backbone.{name}", param) for name, param in backbone.named_parameters()],
            float(get_by_path(cfg, "train.lora_lr", 5e-5)),
        )
    for module_name in ("frame_projector", "set_encoder", "stage_head", "pair_head"):
        module = getattr(raw, module_name, None)
        if module is not None:
            add_group(
                module_name,
                [(f"{module_name}.{name}", param) for name, param in module.named_parameters()],
                float(get_by_path(cfg, "train.head_lr", 3e-4)),
            )
    if not groups:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(groups)


def _make_loss(cfg: dict[str, Any]) -> StagePairStructuredLoss:
    return StagePairStructuredLoss(
        permutation_weight=float(get_by_path(cfg, "loss.permutation_weight", 1.0)),
        stage_weight=float(get_by_path(cfg, "loss.stage_weight", 0.3)),
        pair_weight=float(get_by_path(cfg, "loss.pair_weight", 0.2)),
        consistency_weight=float(get_by_path(cfg, "loss.consistency_weight", 0.0)),
    )


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    frame_chunk_size: int | None = None,
) -> dict[str, torch.Tensor]:
    if "frame_hidden" in batch:
        return model(frame_hidden=batch["frame_hidden"])
    return model(
        inputs=batch["inputs"],
        batch_size=int(batch["batch_size"]),
        anchor_mask=batch.get("anchor_mask"),
        frame_chunk_size=frame_chunk_size,
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    output_dir: str | Path | None,
    baseline_predictions: str | None = None,
    raw_logits_path: str | Path | None = None,
    frame_chunk_size: int | None = None,
    frame_features_path: str | Path | None = None,
) -> dict[str, Any]:
    model.eval()
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress_started = time.perf_counter()
    processed_samples = 0
    ids: list[str] = []
    finals: list[torch.Tensor] = []
    stages: list[torch.Tensor] = []
    pairs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    answers: list[torch.Tensor] = []
    frame_features: list[torch.Tensor] | None = [] if frame_features_path is not None else None
    latencies: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            batch = move_stage_pair_batch_to_device(batch, device)
            start = time.perf_counter()
            outputs = _forward_batch(model, batch, frame_chunk_size=frame_chunk_size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            ids.extend([str(v) for v in batch["id"]])
            finals.append(outputs["final_logits"].detach().cpu())
            stages.append(outputs["stage_logits"].detach().cpu())
            pairs.append(outputs["pair_logits"].detach().cpu())
            targets.append(batch["target_perm_idx"].detach().cpu())
            answers.append(batch["answer"].detach().cpu())
            if frame_features is not None:
                frame_features.append(outputs["frame_hidden"].detach().float().cpu())
            processed_samples += len(batch["id"])
            if processed_samples % 100 == 0 or (total_samples is not None and processed_samples == total_samples):
                total_text = str(total_samples) if total_samples is not None else "?"
                elapsed = time.perf_counter() - progress_started
                print(
                    f"evaluation progress: {processed_samples}/{total_text} "
                    f"samples, elapsed={elapsed:.1f}s",
                    flush=True,
                )
    model_final_t = torch.cat(finals, dim=0).float()
    stage_t = torch.cat(stages, dim=0)
    pair_t = torch.cat(pairs, dim=0)
    target_t = torch.cat(targets, dim=0)
    answer_t = torch.cat(answers, dim=0)
    final_t = canonical_cpu_float32_scores(stage_t, pair_t)
    scorer_max_abs_diff = float((model_final_t - final_t).abs().max().item()) if final_t.numel() else 0.0
    metrics = compute_stage_pair_metrics(final_t, target_t, stage_logits=stage_t, pair_logits=pair_t, answer=answer_t, latencies=latencies)
    metrics["scorer_mode"] = "canonical_cpu_float32"
    metrics["model_vs_canonical_score_max_abs_diff"] = scorer_max_abs_diff
    if raw_logits_path is not None:
        save_raw_stage_pair_logits(
            raw_logits_path,
            ids=ids,
            stage_logits=stage_t,
            pair_logits=pair_t,
            target_perm_idx=target_t,
            answer=answer_t,
        )
    if frame_features_path is not None:
        destination = Path(frame_features_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if frame_features is None:
            raise RuntimeError("frame feature capture was not initialized")
        torch.save(
            {
                "ids": ids,
                "frame_hidden": torch.cat(frame_features, dim=0),
                "frame_chunk_size": frame_chunk_size,
            },
            destination,
        )
    if output_dir is not None:
        write_stage_pair_artifacts(output_dir, ids, final_t, target_t, metrics, stage_logits=stage_t, pair_logits=pair_t)
        if baseline_predictions:
            comparison = compare_with_baseline(Path(output_dir) / "valid_predictions.csv", baseline_predictions)
            if comparison is not None:
                metrics["baseline_comparison"] = comparison
                write_json(Path(output_dir) / "baseline_comparison.json", comparison)
                write_json(Path(output_dir) / "metrics.json", metrics)
    return metrics


def _is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    for key in ("exact_match", "MRR", "top3_accuracy"):
        cand = float(candidate.get(key, 0.0))
        old = float(best.get(key, 0.0))
        if cand > old:
            return True
        if cand < old:
            return False
    return False


def _first_nonfinite_grad_name(model: torch.nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return name
    return None


def _assert_no_nan_grad(model: torch.nn.Module) -> None:
    bad_name = _first_nonfinite_grad_name(model)
    if bad_name is not None:
        raise FloatingPointError(f"NaN/Inf gradient detected in {bad_name}")


def _consistency_logits_if_enabled(model: torch.nn.Module, outputs: dict[str, torch.Tensor], weight: float) -> torch.Tensor | None:
    if weight <= 0:
        return None
    shuffle = torch.tensor([3, 2, 1, 0], device=outputs["frame_hidden"].device)
    shuffled = outputs["frame_hidden"][:, shuffle]
    shuffled_outputs = model(frame_hidden=shuffled)
    return remap_logits_to_canonical(shuffled_outputs["final_logits"], shuffle)


def run_training(cfg: dict[str, Any], mode: str, *, resume: str | None = None) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode}")
    cfg = _configure_mode(cfg, mode)
    if bool(get_by_path(cfg, "retention.enabled", False)) and str(
        get_by_path(cfg, "retention.mode", "")
    ) == "component_safe_v3":
        threshold_audit_path = Path(str(get_by_path(cfg, "retention.threshold_audit", "")))
        if not threshold_audit_path.is_file():
            raise FileNotFoundError(f"Retention v3 threshold audit is missing: {threshold_audit_path}")
        threshold_audit = json.loads(threshold_audit_path.read_text(encoding="utf-8"))
        if threshold_audit.get("status") != "COMPONENT_CACHE_AUDIT_PASS":
            raise RuntimeError("Retention v3 component-cache audit did not pass")
        teacher_cache = Path(str(get_by_path(cfg, "retention.teacher_cache")))
        reference_cache = Path(str(get_by_path(cfg, "retention.reference_cache")))
        if _split_hash(teacher_cache) != threshold_audit.get("teacher_cache_sha256"):
            raise RuntimeError("Retention v3 teacher cache changed after the threshold audit")
        if _split_hash(reference_cache) != threshold_audit.get("reference_cache_sha256"):
            raise RuntimeError("Retention v3 v1 reference cache changed after the threshold audit")
        cfg.setdefault("retention", {})["thresholds"] = threshold_audit["thresholds"]
        cfg["retention"]["teacher_heads_sha256"] = threshold_audit["teacher_identity"]["teacher_heads_sha256"]
        cfg["retention"]["reference_heads_sha256"] = threshold_audit["reference_identity"]["teacher_heads_sha256"]
        coverage_policy = threshold_audit.get("coverage_policy")
        if not isinstance(coverage_policy, dict):
            raise RuntimeError("Retention v3 component-cache audit lacks the preregistered coverage policy")
        cfg["retention"]["weights"] = apply_preregistered_component_coverage(
            dict(get_by_path(cfg, "retention.weights")), coverage_policy
        )
        cfg["retention"]["coverage_policy"] = coverage_policy
    p0_report_path = get_by_path(cfg, "integrity.p0_report", None)
    if p0_report_path:
        p0_report = json.loads(Path(str(p0_report_path)).read_text(encoding="utf-8"))
        if p0_report.get("status") != "P0_CANONICAL_SCORER_PARITY_PASS":
            raise RuntimeError("Training is blocked because the P0 canonical scorer gate did not pass")
    ddp_timeout_seconds = int(get_by_path(cfg, "train.ddp_timeout_seconds", 600))
    if not 1 <= ddp_timeout_seconds <= 86_400:
        raise RuntimeError("train.ddp_timeout_seconds must be in [1, 86400]")
    ddp = _init_distributed(timeout_seconds=ddp_timeout_seconds)
    _force_ddp_single_device_map(cfg, ddp, mode)
    seed = int(get_by_path(cfg, "experiment.seed", 42))
    seed_everything(seed + ddp.rank)
    run_id = _run_id(cfg, mode)
    output_root = Path(str(get_by_path(cfg, "output.dir", "outputs/experiments/qwen3vl_stage_pair")))
    output_dir = output_root / run_id
    best_ckpt = Path(str(get_by_path(cfg, "output.checkpoint_dir", "weights/qwen3vl_stage_pair"))) / run_id / "best"
    last_ckpt = Path(str(get_by_path(cfg, "output.checkpoint_dir", "weights/qwen3vl_stage_pair"))) / run_id / "last"
    if ddp.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_version_report(output_dir / "environment.json")
        write_json(output_dir / "config.json", cfg)
        print(json.dumps({"mode": mode, "run_id": run_id, "distributed": ddp.enabled, "world_size": ddp.world_size}, ensure_ascii=False), flush=True)

    model, processor = _build_model(cfg, mode)
    init_head_from = get_by_path(cfg, "train.init_head_from", None)
    if init_head_from:
        if not is_27b_port_config(cfg):
            raise RuntimeError("--init-head-from is restricted to the 27B Champion port")
        from .qwen35_27b_port import load_migrated_champion_heads

        init_report = load_migrated_champion_heads(str(init_head_from), model)
        if ddp.is_main:
            write_json(output_dir / "head_initialization.json", init_report)
    prompt_fingerprint: dict[str, Any] | None = None
    if processor is not None and int(get_by_path(cfg, "checkpoint.format_version", 1)) in {2, 3}:
        prompt_fingerprint = build_prompt_fingerprint(cfg, processor)
        if ddp.is_main:
            write_prompt_fingerprint(output_dir / "prompt_fingerprint.json", prompt_fingerprint)
    device = ddp.device if ddp.enabled else _model_device(model)
    _move_stage_pair_modules(model, device)
    if mode.startswith("frozen"):
        model.to(device)
    if resume:
        model, _ = load_stage_pair_checkpoint(
            resume,
            model,
            strict=bool(get_by_path(cfg, "checkpoint.strict", True)),
            is_trainable=True,
            cfg=cfg,
            processor=processor,
        )
    if ddp.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[ddp.local_rank],
            output_device=ddp.local_rank,
            find_unused_parameters=bool(get_by_path(cfg, "train.ddp_find_unused_parameters", False)),
        )
    if ddp.is_main:
        report = write_stage_pair_trainable_report(output_dir / "trainable_parameter_report.json", model)
        print(json.dumps({"event": "trainable_parameter_report", **report}, ensure_ascii=False), flush=True)

    train_loader, valid_loader, _ = _make_dataloaders(cfg, mode, ddp, processor)
    optimizer = _build_optimizer(model, cfg)
    gradient_health: GradientHealthMonitor | None = None
    if bool(get_by_path(cfg, "vision_merger_lora.enabled", False)):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("E2 requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or ddp.world_size != 1:
            raise RuntimeError(
                "E2 requires exactly one visible CUDA device and distributed world size 1: "
                f"cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}, "
                f"world_size={ddp.world_size}"
            )
        health_path = Path(
            os.environ.get("GRADIENT_HEALTH_OUTPUT", str(output_dir / "gradient_health.json"))
        )
        gradient_health = GradientHealthMonitor(unwrap_model(model), health_path)
    epochs = int(get_by_path(cfg, "train.epochs_frozen" if mode.startswith("frozen") else "train.epochs", 20 if mode.startswith("frozen") else 4))
    configured_grad_accum = int(get_by_path(cfg, "train.gradient_accumulation_steps", 1 if mode.startswith("frozen") else 8))
    grad_accum = max(1, math.ceil(configured_grad_accum / ddp.world_size)) if ddp.enabled and bool(get_by_path(cfg, "train.ddp_adjust_gradient_accumulation", True)) else configured_grad_accum
    total_steps = max(1, math.ceil(len(train_loader) / grad_accum) * epochs)
    head_warmup_fraction = float(get_by_path(cfg, "train.head_warmup_fraction", 0.0))
    if not 0.0 <= head_warmup_fraction < 1.0:
        raise RuntimeError("train.head_warmup_fraction must be in [0,1)")
    if head_warmup_fraction > 0.0 and ddp.enabled and not bool(
        get_by_path(cfg, "train.ddp_find_unused_parameters", False)
    ):
        raise RuntimeError("27B head warm-up under DDP requires ddp_find_unused_parameters=true")
    head_warmup_steps = int(math.ceil(total_steps * head_warmup_fraction))
    retention_enabled = bool(get_by_path(cfg, "retention.enabled", False))
    retention_v3 = retention_enabled and str(get_by_path(cfg, "retention.mode", "")) == "component_safe_v3"
    scheduler: Any
    if retention_enabled:
        if mode != "qlora_stage_pair" or not is_27b_port_config(cfg):
            raise RuntimeError("Champion retention is restricted to the 27B QLoRA model")
        if retention_v3:
            if abs(head_warmup_fraction - 0.10) > 1e-12:
                raise RuntimeError("Retention v3 requires the v1 head_warmup_fraction=0.10")
            scheduler = _scheduler(optimizer, total_steps, float(get_by_path(cfg, "train.warmup_ratio", 0.05)))
        else:
            if head_warmup_fraction != 0.0:
                raise RuntimeError("Legacy Champion retention schedule requires train.head_warmup_fraction=0")
            scheduler = ChampionRetentionLRScheduler(optimizer, total_steps, cfg)
        if ddp.is_main:
            write_json(
                output_dir / "champion_retention_contract.json",
                component_safe_contract(cfg) if retention_v3 else retention_contract(cfg, scheduler),
            )
    else:
        scheduler = _scheduler(optimizer, total_steps, float(get_by_path(cfg, "train.warmup_ratio", 0.05)))
    start_epoch = 1
    resume_next_step = 1
    global_step = 0
    resume_rng_state: dict[str, Any] | None = None
    if resume and bool(get_by_path(cfg, "train.resume_training_state", False)):
        training_state_path = Path(resume) / "training_state.pt"
        if not training_state_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint lacks training_state.pt: {resume}")
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        required_state = {"optimizer", "scheduler", "training_progress"}
        if not isinstance(training_state, dict) or not required_state.issubset(training_state):
            raise RuntimeError("Branch resume training state is incomplete")
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        progress_state = training_state["training_progress"]
        start_epoch = int(progress_state["epoch"])
        resume_next_step = int(progress_state["next_step_in_epoch"])
        global_step = int(progress_state["global_step"])
        rng_path = _branch_rng_dir(resume) / f"rank_{ddp.rank:02d}.pt"
        if not rng_path.is_file():
            raise FileNotFoundError(f"Rank RNG state is missing: {rng_path}")
        resume_rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        if ddp.is_main:
            write_json(
                output_dir / "branch_resume_contract.json",
                {
                    "checkpoint": str(Path(resume).resolve()),
                    "global_step": global_step,
                    "epoch": start_epoch,
                    "next_step_in_epoch": resume_next_step,
                    "world_size": ddp.world_size,
                },
            )
    amp_dtype, scaler_enabled = _amp_dtype_and_scaler_enabled(cfg)
    use_amp = torch.cuda.is_available() and amp_dtype in {torch.bfloat16, torch.float16}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and scaler_enabled)
    loss_fn = _make_loss(cfg).to(device)
    consistency_weight = float(get_by_path(cfg, "loss.consistency_weight", 0.0))
    grad_clip = float(get_by_path(cfg, "train.grad_clip", 1.0))
    skip_nonfinite_grad_steps = bool(get_by_path(cfg, "train.skip_nonfinite_grad_steps", False))
    max_consecutive_nonfinite = int(get_by_path(cfg, "train.max_consecutive_nonfinite_grad_steps", 3))
    consecutive_nonfinite = 0
    patience = int(get_by_path(cfg, "train.early_stopping_patience", 4 if mode.startswith("frozen") else 2))
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    extra = {
        "git": _git_report(),
        "splits": {
            "train_split": str(get_by_path(cfg, "data.train_split")),
            "valid_split": str(get_by_path(cfg, "data.valid_split")),
            "train_hash": _split_hash(str(get_by_path(cfg, "data.train_split"))),
            "valid_hash": _split_hash(str(get_by_path(cfg, "data.valid_split"))),
        },
        "mode": mode,
        "run_id": run_id,
    }

    gradient_contract_checked = False
    last_retention_phase: str | None = None
    fork_completed = False
    branch_first_forward_recorded = False
    fork_after_head_warmup = bool(get_by_path(cfg, "train.fork_after_head_warmup", False))
    fork_ckpt = Path(str(get_by_path(cfg, "output.checkpoint_dir", "weights/qwen3vl_stage_pair"))) / run_id / "fork_10pct"
    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and step < resume_next_step:
                continue
            if resume_rng_state is not None:
                _restore_rank_rng_state(resume_rng_state, device)
                resume_rng_state = None
            raw_for_schedule = unwrap_model(model)
            retention_phase: str | None = None
            if retention_enabled and not retention_v3:
                retention_phase = scheduler.apply(global_step)
                freeze_backbone = retention_phase != "joint_qlora"
                raw_for_schedule.set_backbone_forward_frozen(freeze_backbone)
                if freeze_backbone and getattr(raw_for_schedule, "backbone", None) is not None:
                    raw_for_schedule.backbone.eval()
                elif getattr(raw_for_schedule, "backbone", None) is not None:
                    raw_for_schedule.backbone.train()
                if retention_phase != last_retention_phase:
                    if ddp.is_main:
                        print(
                            json.dumps(
                                {
                                    "event": "retention_phase_transition",
                                    "phase": retention_phase,
                                    "global_step": global_step,
                                    "learning_rates": {
                                        str(group.get("group_name")): float(group["lr"])
                                        for group in optimizer.param_groups
                                    },
                                }
                            ),
                            flush=True,
                        )
                    last_retention_phase = retention_phase
            elif hasattr(raw_for_schedule, "set_backbone_forward_frozen"):
                raw_for_schedule.set_backbone_forward_frozen(global_step < head_warmup_steps)
            batch = move_stage_pair_batch_to_device(batch, device)
            sync_step = step % grad_accum == 0 or step == len(train_loader)
            sync_context = model.no_sync() if ddp.enabled and not sync_step else contextlib.nullcontext()
            with sync_context:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    outputs = _forward_batch(model, batch)
                    if resume and not branch_first_forward_recorded:
                        first_forward_payload = {
                            "resume_checkpoint": str(Path(resume).resolve()),
                            "global_step": global_step,
                            "epoch": epoch,
                            "step": step,
                            "ids": [str(value) for value in batch["id"]],
                            "output_sha256": _first_forward_sha256(outputs, batch["id"]),
                        }
                        if retention_v3 and ddp.is_main:
                            control_forward_path = Path(
                                str(get_by_path(cfg, "retention.control_first_forward", ""))
                            )
                            if not control_forward_path.is_file():
                                raise FileNotFoundError(
                                    f"K0 control first-forward artifact is missing: {control_forward_path}"
                                )
                            control_forward = json.loads(control_forward_path.read_text(encoding="utf-8"))
                            for key in ("global_step", "epoch", "step", "ids", "output_sha256"):
                                if first_forward_payload[key] != control_forward.get(key):
                                    raise RuntimeError(
                                        f"C0/K0 first-forward parity failed for {key}: "
                                        f"control={control_forward.get(key)!r} candidate={first_forward_payload[key]!r}"
                                    )
                        if ddp.is_main:
                            write_json(
                                output_dir / "branch_first_forward.json",
                                first_forward_payload,
                            )
                        branch_first_forward_recorded = True
                    consistency_logits = _consistency_logits_if_enabled(model, outputs, consistency_weight)
                    loss_out = loss_fn(outputs, batch["target_perm_idx"], batch["answer"], consistency_logits=consistency_logits)
                    if retention_v3:
                        retention_out = None
                        component_out = component_safe_retention_loss(
                            outputs,
                            batch,
                            cfg,
                            base_loss=loss_out.loss,
                            progress=float(global_step) / float(max(total_steps, 1)),
                        )
                        total_loss = loss_out.loss + component_out.loss
                    else:
                        component_out = None
                        retention_out = champion_retention_loss(outputs, batch, cfg)
                        total_loss = loss_out.loss + retention_out.loss
                    loss = total_loss / grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"NaN/Inf loss at epoch={epoch} step={step} rank={ddp.rank}")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if (
                not gradient_contract_checked
                and mode == "qlora_stage_pair"
                and global_step >= head_warmup_steps
                and (not retention_enabled or retention_v3 or retention_phase == "joint_qlora")
            ):
                raw_model = unwrap_model(model)
                if isinstance(getattr(getattr(raw_model, "backbone", None), "_stage_pair_lora_target_manifest", None), list):
                    gradient_report = validate_gradient_contract(raw_model)
                    gradient_contract_checked = True
                    if ddp.is_main:
                        print(
                            json.dumps({"event": "gradient_contract", **gradient_report}, ensure_ascii=False),
                            flush=True,
                        )
            running += float(total_loss.detach().cpu())
            seen += 1
            if sync_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                bad_grad_name = _first_nonfinite_grad_name(model)
                if bad_grad_name is not None:
                    if not skip_nonfinite_grad_steps:
                        raise FloatingPointError(f"NaN/Inf gradient detected in {bad_grad_name}")
                    consecutive_nonfinite += 1
                    if ddp.is_main:
                        print(
                            json.dumps(
                                {
                                    "event": "skip_nonfinite_grad_step",
                                    "epoch": epoch,
                                    "step": step,
                                    "global_step": global_step,
                                    "parameter": bad_grad_name,
                                    "consecutive": consecutive_nonfinite,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if consecutive_nonfinite >= max_consecutive_nonfinite:
                        raise FloatingPointError(
                            f"NaN/Inf gradients repeated {consecutive_nonfinite} consecutive optimizer steps; "
                            f"last parameter={bad_grad_name}"
                        )
                    continue
                consecutive_nonfinite = 0
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                pending_health = (
                    gradient_health.capture_before_step(global_step + 1)
                    if gradient_health is not None
                    else {}
                )
                optimizer_step_completed = True
                if scaler.is_enabled():
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_completed = float(scaler.get_scale()) >= scale_before
                else:
                    optimizer.step()
                if not optimizer_step_completed:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if gradient_health is not None:
                    gradient_health.capture_after_step(pending_health)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if fork_after_head_warmup and global_step == head_warmup_steps:
                    captured_rng = _capture_rank_rng_state(device)
                    progress = {
                        "epoch": epoch,
                        "next_step_in_epoch": step + 1,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "head_warmup_steps": head_warmup_steps,
                        "world_size": ddp.world_size,
                    }
                    if ddp.is_main:
                        save_stage_pair_checkpoint(
                            fork_ckpt,
                            model,
                            cfg,
                            {"status": "SHARED_WARMUP_FORK", "global_step": global_step},
                            processor=processor,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            extra=extra,
                            prompt_fingerprint=prompt_fingerprint,
                            training_progress=progress,
                        )
                        _branch_rng_dir(fork_ckpt).mkdir(parents=True, exist_ok=False)
                    _barrier(ddp)
                    torch.save(captured_rng, _branch_rng_dir(fork_ckpt) / f"rank_{ddp.rank:02d}.pt")
                    _barrier(ddp)
                    fork_completed = True
                    break
                if ddp.is_main and global_step % int(get_by_path(cfg, "train.log_interval_steps", 20)) == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "rank0_loss": running / max(seen, 1),
                                "retention_phase": retention_phase,
                                "retention_active_samples": (
                                    sum(component_out.active_counts.values())
                                    if component_out is not None
                                    else retention_out.active_samples
                                ),
                                "retention_schedule_scale": (
                                    component_out.schedule_scale if component_out is not None else None
                                ),
                                "retention_active_components": (
                                    component_out.active_counts if component_out is not None else None
                                ),
                                "learning_rates": {
                                    str(group.get("group_name")): float(group["lr"])
                                    for group in optimizer.param_groups
                                },
                            }
                        ),
                        flush=True,
                    )

        if fork_completed:
            break
        loss_count = torch.tensor([running, float(seen)], device=device)
        if ddp.enabled:
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        train_loss = float(loss_count[0].item() / max(loss_count[1].item(), 1.0))
        stop_training = False
        if ddp.is_main:
            if valid_loader is None:
                raise RuntimeError("valid_loader missing on rank0")
            epoch_output_dir = output_dir / "valid_a" / f"epoch_{epoch:03d}"
            baseline_predictions = str(get_by_path(cfg, "baseline.valid_predictions", "")) or None
            if retention_v3:
                control_summary_path = Path(str(get_by_path(cfg, "retention.control_summary", "")))
                baseline_predictions = str(
                    control_summary_path.parent
                    / "valid_a"
                    / f"epoch_{epoch:03d}"
                    / "valid_predictions.csv"
                )
            metrics = evaluate_model(
                unwrap_model(model),
                valid_loader,
                device=device,
                output_dir=epoch_output_dir,
                baseline_predictions=baseline_predictions,
            )
            metrics["epoch"] = epoch
            metrics["global_step"] = global_step
            metrics["train_loss"] = train_loss
            retention_kill_reasons: list[str] = []
            if retention_v3:
                control_summary_path = Path(str(get_by_path(cfg, "retention.control_summary", "")))
                if not control_summary_path.is_file():
                    raise FileNotFoundError(f"K0 control summary is missing: {control_summary_path}")
                control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
                control_epoch = next(
                    (row for row in control_summary.get("history", []) if int(row.get("epoch", -1)) == epoch),
                    None,
                )
                if control_epoch is None:
                    raise RuntimeError(f"C0 control lacks epoch {epoch} metrics")
                if epoch == 1:
                    if int(metrics["correct_count"]) < int(control_epoch["correct_count"]) - 10:
                        retention_kill_reasons.append("epoch1_correct_below_c0_minus_10")
                    stage_ratio = float(metrics["stage_component_score_std"]) / max(
                        float(control_epoch["stage_component_score_std"]), 1e-12
                    )
                    margin_ratio = float(metrics["mean_top1_top2_margin"]) / max(
                        float(control_epoch["mean_top1_top2_margin"]), 1e-12
                    )
                    if stage_ratio < 0.92:
                        retention_kill_reasons.append("epoch1_stage_std_ratio_below_0p92")
                    if margin_ratio < 0.90:
                        retention_kill_reasons.append("epoch1_margin_ratio_below_0p90")
                else:
                    paired = metrics.get("baseline_comparison")
                    if not isinstance(paired, dict) or int(paired.get("common_count", 0)) != len(valid_loader.dataset):
                        raise RuntimeError(f"K0 epoch {epoch} lacks a complete paired C0 comparison")
                    fixed = int(paired["old_wrong_new_correct"])
                    broken = int(paired["old_correct_new_wrong"])
                    if fixed - broken <= -8:
                        retention_kill_reasons.append("paired_net_le_minus_8")
                    if int(metrics["stage_only_correct_count"]) <= int(control_epoch["stage_only_correct_count"]) - 10:
                        retention_kill_reasons.append("stage_only_below_c0_by_10")
                    if int(metrics["high_margin_wrong_count"]) >= int(control_epoch["high_margin_wrong_count"]) + 8:
                        retention_kill_reasons.append("high_margin_wrong_above_c0_by_8")
                    swap_34 = int(metrics["adjacent_swap_by_stage"]["3-4"])
                    control_swap_34 = int(control_epoch["adjacent_swap_by_stage"]["3-4"])
                    if swap_34 > control_swap_34 and int(metrics["correct_count"]) <= int(control_epoch["correct_count"]):
                        retention_kill_reasons.append("swap_3_4_worse_without_exact_gain")
                metrics["retention_kill_gate"] = {
                    "control_epoch": epoch,
                    "reasons": retention_kill_reasons,
                    "stop": bool(retention_kill_reasons),
                }
            history.append(metrics)
            print(json.dumps({"epoch": epoch, "valid_exact": metrics["exact_match"], "stage_accuracy": metrics["stage_accuracy"]}, ensure_ascii=False), flush=True)
            save_stage_pair_checkpoint(
                last_ckpt,
                model,
                cfg,
                metrics,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                extra=extra,
                prompt_fingerprint=prompt_fingerprint,
            )
            if _is_better(metrics, best_metrics):
                best_metrics = dict(metrics)
                epochs_without_improvement = 0
                save_stage_pair_checkpoint(
                    best_ckpt,
                    model,
                    cfg,
                    best_metrics,
                    processor=processor,
                    extra=extra,
                    minimal=True,
                    prompt_fingerprint=prompt_fingerprint,
                )
            else:
                epochs_without_improvement += 1
                stop_training = patience > 0 and epochs_without_improvement >= patience
            stop_training = stop_training or bool(retention_kill_reasons)
        if _broadcast_stop_flag(stop_training, ddp):
            break

    if not ddp.is_main:
        return {"mode": mode, "rank": ddp.rank, "status": "worker_complete"}
    if gradient_health is not None:
        gradient_health.assert_complete()
    summary = {
        "mode": mode,
        "run_id": run_id,
        "status": "SHARED_WARMUP_FORK_READY" if fork_completed else "TRAINING_COMPLETE",
        "fork_checkpoint": str(fork_ckpt) if fork_completed else None,
        "best": best_metrics or {},
        "history": history,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "champion_retention": (
            None
            if not retention_enabled
            else component_safe_contract(cfg)
            if retention_v3
            else retention_contract(cfg, scheduler)
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--source", choices=["base", "existing_lora"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--init-head-from", default=None)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--valid-split", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--base-model-revision", default=None)
    parser.add_argument("--fork-after-head-warmup", action="store_true")
    parser.add_argument("--resume-training-state", action="store_true")
    args = parser.parse_args()
    cfg = _apply_cli_overrides(load_config(args.config), args)
    try:
        result = run_training(cfg, args.mode, resume=args.resume)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
