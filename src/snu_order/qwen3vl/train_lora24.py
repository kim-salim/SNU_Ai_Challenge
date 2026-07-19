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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import read_csv_rows, write_json
from snu_order.utils.seed import seed_everything

from .checkpoint import load_lora24_checkpoint, save_lora24_checkpoint, unwrap_model
from .collator import Qwen3VLCollator, move_batch_to_device
from .dataset import Qwen3VLFrameOrderDataset, fixed_subset_indices
from .lockbox import guard_not_lockbox_path
from .metrics24 import compute_metrics_from_logits, write_eval_artifacts
from .modeling_lora24 import (
    build_qwen3vl_lora24_model,
    dump_version_report,
    trainable_parameter_report,
    write_trainable_report,
)
from .structured_loss import StructuredPermutationLoss


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _init_distributed(timeout_seconds: int = 600) -> DistributedState:
    if timeout_seconds <= 0:
        raise ValueError("DDP timeout must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistributedState(False, 0, 0, 1, device)
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA for this Qwen3-VL LoRA training path")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout = timedelta(seconds=int(timeout_seconds))
        try:
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timeout,
            )
        except TypeError:
            dist.init_process_group(backend="nccl", timeout=timeout)
    return DistributedState(True, rank, local_rank, world_size, torch.device(f"cuda:{local_rank}"))


def _barrier(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        try:
            dist.barrier(device_ids=[state.local_rank])
        except TypeError:
            dist.barrier()


def _broadcast_stop_flag(stop: bool, state: DistributedState) -> bool:
    if not state.enabled:
        return bool(stop)
    flag = torch.tensor([1 if stop else 0], device=state.device, dtype=torch.int64)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = deepcopy(cfg)
    if args.output_dir:
        out.setdefault("output", {})["dir"] = args.output_dir
    if args.seed is not None:
        out.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.subset_ratio is not None:
        out.setdefault("train", {})["subset_ratio"] = float(args.subset_ratio)
    if args.max_samples is not None:
        out.setdefault("train", {})["subset_max_samples"] = int(args.max_samples)
    return out


def _git_report() -> dict[str, Any]:
    report = {"commit": None, "dirty": None}
    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True)
        report["dirty"] = bool(status.strip())
    except Exception:
        pass
    return report


def _split_hash(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def _make_loss(cfg: dict[str, Any]) -> StructuredPermutationLoss:
    return StructuredPermutationLoss(
        permutation_weight=float(get_by_path(cfg, "loss.permutation_weight", 1.0)),
        pairwise_marginal_weight=float(get_by_path(cfg, "loss.pairwise_marginal_weight", 0.2)),
        position_marginal_weight=float(get_by_path(cfg, "loss.position_marginal_weight", 0.1)),
        label_smoothing=float(get_by_path(cfg, "loss.label_smoothing", 0.05)),
    )


def _amp_dtype_and_scaler_enabled(cfg: dict[str, Any]) -> tuple[torch.dtype, bool]:
    precision = str(get_by_path(cfg, "train.mixed_precision", "bf16")).lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16, False
    if precision in {"fp16", "float16", "half"}:
        return torch.float16, bool(get_by_path(cfg, "train.use_grad_scaler", True))
    if precision in {"fp32", "float32", "none"}:
        return torch.float32, False
    raise ValueError(f"Unsupported train.mixed_precision: {precision}")


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> Any:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _decay_groups(named_params: list[tuple[str, torch.nn.Parameter]], lr: float, weight_decay: float) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for name, parameter in named_params:
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if lowered.endswith("bias") or "norm" in lowered or "layernorm" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    weight_decay = float(get_by_path(cfg, "train.weight_decay", 0.01))
    groups = []
    groups.extend(
        _decay_groups(
            [(f"backbone.{name}", param) for name, param in model.backbone.named_parameters()],
            lr=float(get_by_path(cfg, "train.lora_lr", 5e-5)),
            weight_decay=weight_decay,
        )
    )
    groups.extend(
        _decay_groups(
            [(f"classifier.{name}", param) for name, param in model.classifier.named_parameters()],
            lr=float(get_by_path(cfg, "train.classifier_lr", 3e-4)),
            weight_decay=weight_decay,
        )
    )
    if not groups:
        raise ValueError("No trainable parameters found for optimizer")
    return torch.optim.AdamW(groups)


def build_datasets(cfg: dict[str, Any], mode: str) -> tuple[Qwen3VLFrameOrderDataset, Qwen3VLFrameOrderDataset | None]:
    train_split = str(get_by_path(cfg, "data.train_split"))
    valid_split = str(get_by_path(cfg, "data.valid_split"))
    image_root = str(get_by_path(cfg, "data.image_root", "data/raw"))
    guard_not_lockbox_path(train_split, purpose="training")
    guard_not_lockbox_path(valid_split, purpose="early stopping")
    seed = int(get_by_path(cfg, "experiment.seed", 42))
    train_rows = read_csv_rows(train_split)
    sample_indices = None
    max_samples = None
    augment = bool(get_by_path(cfg, "data.train_permutation_augmentation", True))
    if mode == "overfit64":
        sample_indices = fixed_subset_indices(len(train_rows), 64, seed)
        augment = bool(get_by_path(cfg, "train.overfit64_augmentation", False))
    elif mode in {"subset", "frozen_probe"}:
        subset_ratio = float(get_by_path(cfg, "train.subset_ratio", 0.2))
        subset_max = int(get_by_path(cfg, "train.subset_max_samples", 1024))
        size = min(subset_max, max(1, int(len(train_rows) * subset_ratio)))
        sample_indices = fixed_subset_indices(len(train_rows), size, seed)
    elif mode != "full":
        raise ValueError(f"Unsupported mode: {mode}")
    if mode != "overfit64" and int(get_by_path(cfg, "train.subset_max_samples", -1)) > 0 and mode == "subset":
        max_samples = None
    train_dataset = Qwen3VLFrameOrderDataset(
        train_split,
        image_root,
        training=True,
        augment_permutation=augment,
        permutation_probability=float(get_by_path(cfg, "data.permutation_probability", 1.0)),
        seed=seed,
        max_samples=max_samples,
        sample_indices=sample_indices,
    )
    if mode == "overfit64":
        valid_dataset = Qwen3VLFrameOrderDataset(
            train_split,
            image_root,
            training=False,
            augment_permutation=False,
            seed=seed,
            sample_indices=sample_indices,
        )
    else:
        valid_dataset = Qwen3VLFrameOrderDataset(valid_split, image_root, training=False, augment_permutation=False, seed=seed)
    return train_dataset, valid_dataset


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    output_dir: str | Path | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_ids: list[str] = []
    latencies: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            start = time.perf_counter()
            logits = model(**batch["inputs"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            all_logits.append(logits.detach().cpu())
            all_targets.append(batch["target_perm_idx"].detach().cpu())
            all_ids.extend([str(v) for v in batch["id"]])
    logits_tensor = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, 24))
    target_tensor = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.long)
    metrics = compute_metrics_from_logits(logits_tensor, target_tensor, latencies=latencies)
    if profile:
        metrics["samples_per_sec"] = metrics["sample_count"] / max(sum(latencies), 1e-9)
    if output_dir is not None:
        write_eval_artifacts(output_dir, all_ids, logits_tensor, target_tensor, metrics)
        write_json(Path(output_dir) / "environment.json", {"cuda_available": torch.cuda.is_available()})
    return metrics


def _is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    keys = ("exact_match", "MRR", "top3_accuracy")
    for key in keys:
        cand = float(candidate.get(key, 0.0))
        old = float(best.get(key, 0.0))
        if cand > old:
            return True
        if cand < old:
            return False
    return False


def _assert_no_nan_grad(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"NaN/Inf gradient detected in {name}")


def run_training(cfg: dict[str, Any], mode: str, *, resume: str | None = None) -> dict[str, Any]:
    ddp = _init_distributed()
    seed = int(get_by_path(cfg, "experiment.seed", 42))
    seed_everything(seed + ddp.rank)
    output_dir = Path(str(get_by_path(cfg, "output.dir", "outputs/experiments/qwen3vl_8b_lora24")))
    if ddp.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_version_report(output_dir / "environment.json")
        print(
            json.dumps(
                {
                    "distributed_initialized": ddp.enabled,
                    "rank": ddp.rank,
                    "local_rank": ddp.local_rank,
                    "world_size": ddp.world_size,
                    "device": str(ddp.device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    frozen_probe = mode == "frozen_probe"
    model, processor = build_qwen3vl_lora24_model(cfg, frozen_probe=frozen_probe)
    device = ddp.device if ddp.enabled else _model_device(model)
    model.classifier.to(device)
    if resume:
        model, _ = load_lora24_checkpoint(resume, model, strict=False, is_trainable=True)

    report = trainable_parameter_report(model)
    if ddp.is_main:
        write_trainable_report(output_dir / "trainable_parameters.json", model)
        print(
            json.dumps(
                {
                    "trainable_parameters": report["trainable"],
                    "total_parameters": report["total"],
                    "ratio": report["ratio"],
                    "distributed": {
                        "enabled": ddp.enabled,
                        "world_size": ddp.world_size,
                    },
                },
                indent=2,
            ),
            flush=True,
        )

    train_dataset, valid_dataset = build_datasets(cfg, mode)
    collator = Qwen3VLCollator(processor)
    batch_size = int(get_by_path(cfg, "train.micro_batch_size", 1))
    if batch_size != 1:
        raise ValueError("Qwen3 LoRA24 first-pass implementation supports micro_batch_size=1")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True, seed=seed)
        if ddp.enabled
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=int(get_by_path(cfg, "train.num_workers", 0)),
        worker_init_fn=_seed_worker,
        generator=None if train_sampler is not None else generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0) if ddp.is_main else None

    optimizer = build_optimizer(model, cfg)
    if ddp.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[ddp.local_rank],
            output_device=ddp.local_rank,
            find_unused_parameters=bool(get_by_path(cfg, "train.ddp_find_unused_parameters", False)),
        )
    epochs = int(get_by_path(cfg, "train.overfit64_epochs", 20)) if mode == "overfit64" else int(get_by_path(cfg, "train.epochs", 4))
    configured_grad_accum = int(get_by_path(cfg, "train.gradient_accumulation_steps", 16))
    if ddp.enabled and bool(get_by_path(cfg, "train.ddp_adjust_gradient_accumulation", True)):
        grad_accum = max(1, math.ceil(configured_grad_accum / ddp.world_size))
    else:
        grad_accum = configured_grad_accum
    total_steps = max(1, math.ceil(len(train_loader) / grad_accum) * epochs)
    scheduler = _scheduler(optimizer, total_steps, float(get_by_path(cfg, "train.warmup_ratio", 0.05)))
    loss_fn = _make_loss(cfg).to(device)
    amp_dtype, scaler_enabled = _amp_dtype_and_scaler_enabled(cfg)
    use_amp = torch.cuda.is_available() and amp_dtype in {torch.bfloat16, torch.float16}
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and scaler_enabled)
    grad_clip = float(get_by_path(cfg, "train.grad_clip", 1.0))
    patience = int(get_by_path(cfg, "train.early_stopping_patience", 2))
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    global_step = 0

    best_ckpt = Path(str(get_by_path(cfg, "output.best_checkpoint", "weights/qwen3vl_lora24/best")))
    last_ckpt = Path(str(get_by_path(cfg, "output.last_checkpoint", "weights/qwen3vl_lora24/last")))
    extra = {
        "git": _git_report(),
        "splits": {
            "train_split": str(get_by_path(cfg, "data.train_split")),
            "valid_split": str(get_by_path(cfg, "data.valid_split")),
            "train_hash": _split_hash(str(get_by_path(cfg, "data.train_split"))),
            "valid_hash": _split_hash(str(get_by_path(cfg, "data.valid_split"))),
        },
        "mode": mode,
        "distributed": {
            "enabled": ddp.enabled,
            "world_size": ddp.world_size,
            "configured_gradient_accumulation_steps": configured_grad_accum,
            "per_rank_gradient_accumulation_steps": grad_accum,
        },
        "trainable_parameter_report": report,
    }

    try:
        for epoch in range(1, epochs + 1):
            train_dataset.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            seen = 0
            log_interval = int(get_by_path(cfg, "train.log_interval_steps", 0))
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                sync_step = step % grad_accum == 0 or step == len(train_loader)
                sync_context = model.no_sync() if ddp.enabled and not sync_step else contextlib.nullcontext()
                with sync_context:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        try:
                            logits = model(**batch["inputs"])
                        except Exception as exc:
                            input_shapes = {
                                key: tuple(value.shape)
                                for key, value in batch["inputs"].items()
                                if torch.is_tensor(value)
                            }
                            raise RuntimeError(
                                f"Qwen3 forward failed at epoch={epoch} step={step} "
                                f"rank={ddp.rank} input_shapes={input_shapes}"
                            ) from exc
                        loss_out = loss_fn(logits, batch["target_perm_idx"], batch["answer"])
                        loss = loss_out.loss / grad_accum
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"NaN/Inf loss at epoch={epoch} step={step} rank={ddp.rank}")
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                running += float(loss_out.loss.detach().cpu())
                seen += 1
                if sync_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    _assert_no_nan_grad(model)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if ddp.is_main and log_interval > 0 and global_step % log_interval == 0:
                        print(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "step": step,
                                    "global_step": global_step,
                                    "rank0_running_loss": running / max(seen, 1),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
            loss_count = torch.tensor([running, float(seen)], device=device)
            if ddp.enabled:
                dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            train_loss = float(loss_count[0].item() / max(loss_count[1].item(), 1.0))

            stop_training = False
            if ddp.is_main:
                eval_dir = output_dir / ("train_overfit64" if mode == "overfit64" else "valid_a")
                if valid_loader is None:
                    raise RuntimeError("valid_loader is unexpectedly None on main process")
                metrics = evaluate_model(
                    unwrap_model(model),
                    valid_loader,
                    device=device,
                    output_dir=eval_dir,
                    profile=bool(get_by_path(cfg, "eval.profile", False)),
                )
                metrics["epoch"] = epoch
                metrics["global_step"] = global_step
                metrics["train_loss"] = train_loss
                history.append(metrics)
                print(json.dumps({"epoch": epoch, "valid_exact": metrics["exact_match"], "MRR": metrics["MRR"]}, indent=2), flush=True)
                save_lora24_checkpoint(last_ckpt, model, processor, cfg, metrics, optimizer=optimizer, scheduler=scheduler, extra=extra)
                if _is_better(metrics, best_metrics):
                    best_metrics = dict(metrics)
                    epochs_without_improvement = 0
                    best_payload = dict(best_metrics)
                    best_payload["valid_a_exact_match"] = float(best_metrics["exact_match"])
                    save_lora24_checkpoint(best_ckpt, model, processor, cfg, best_payload, extra=extra, minimal=True)
                else:
                    epochs_without_improvement += 1
                    stop_training = mode != "overfit64" and patience > 0 and epochs_without_improvement >= patience
            if _broadcast_stop_flag(stop_training, ddp):
                break
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA OOM during Qwen3 LoRA24 training. Reduce image pixels or disable bf16 fallback only explicitly.") from exc

    if not ddp.is_main:
        return {"mode": mode, "distributed_rank": ddp.rank, "status": "worker_complete"}

    best_metrics = best_metrics or {}
    valid_a_threshold = float(get_by_path(cfg, "lockbox.min_valid_a_accuracy", 0.33))
    candidate_status = "READY_FOR_LOCKBOX" if float(best_metrics.get("exact_match", 0.0)) >= valid_a_threshold else "VALID_A_BELOW_LOCKBOX_THRESHOLD"
    if mode == "overfit64":
        candidate_status = "OVERFIT64_PASS" if float(best_metrics.get("exact_match", 0.0)) >= 0.90 else "OVERFIT64_FAIL"
    summary = {
        "mode": mode,
        "best": best_metrics,
        "history": history,
        "candidate_status": candidate_status,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }
    write_json(output_dir / f"{mode}_summary.json", summary)
    if candidate_status == "READY_FOR_LOCKBOX":
        print("valid_a threshold passed. Run valid_b only after explicit approval:")
        print("bash scripts/run_qwen3vl_lora_valid_b_lockbox.sh --unlock-valid-b")
    elif mode == "overfit64" and candidate_status == "OVERFIT64_FAIL":
        print("overfit64 sanity failed; do not start full training automatically.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["overfit64", "subset", "full", "frozen_probe"], required=True)
    parser.add_argument("--subset-ratio", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = _apply_cli_overrides(load_config(args.config), args)
    result = run_training(cfg, args.mode, resume=args.resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
