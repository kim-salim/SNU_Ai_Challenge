from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from snu_order.data.dataset import FeatureCacheDataset
from snu_order.features.cache import load_feature_cache, validate_feature_cache
from snu_order.models.losses import PermPairLoss
from snu_order.pipeline.evaluate import evaluate_cache, write_eval_outputs
from snu_order.pipeline.model_io import build_modules, save_checkpoint
from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.device import get_device
from snu_order.utils.io import write_json
from snu_order.utils.seed import seed_everything


def _make_scheduler(optimizer: Any, total_steps: int, warmup_ratio: float) -> Any:
    import torch

    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg: dict[str, Any]) -> dict[str, float]:
    import torch
    from torch.utils.data import DataLoader

    seed = int(get_by_path(cfg, "experiment.seed", 42))
    seed_everything(seed)
    device = get_device()

    feature_dir = Path(str(get_by_path(cfg, "data.feature_dir", "data/features/siglip2_base_224")))
    train_cache = load_feature_cache(feature_dir / "train.npz")
    valid_cache = load_feature_cache(feature_dir / "valid.npz")
    validate_feature_cache(train_cache, require_labels=True)
    validate_feature_cache(valid_cache, require_labels=True)

    embedding_dim = int(train_cache["text_emb"].shape[1])
    quality_dim = int(train_cache["quality"].shape[2])
    projector, ranker, pairwise = build_modules(embedding_dim, quality_dim, cfg, device)

    dataset = FeatureCacheDataset(train_cache, require_labels=True)
    batch_size = int(get_by_path(cfg, "train.batch_size", 128))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    params = list(projector.parameters()) + list(ranker.parameters())
    if pairwise is not None:
        params += list(pairwise.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(get_by_path(cfg, "train.lr", 3e-4)),
        weight_decay=float(get_by_path(cfg, "train.weight_decay", 1e-2)),
    )
    epochs = int(get_by_path(cfg, "train.epochs", 20))
    scheduler = _make_scheduler(
        optimizer,
        total_steps=max(1, len(loader) * epochs),
        warmup_ratio=float(get_by_path(cfg, "train.warmup_ratio", 0.05)),
    )
    loss_fn = PermPairLoss(
        pair_aux_weight=float(get_by_path(cfg, "model.pairwise_aux.weight", 0.3)),
        label_smoothing=float(get_by_path(cfg, "train.label_smoothing", 0.05)),
    )
    use_amp = bool(get_by_path(cfg, "train.amp", True)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = float(get_by_path(cfg, "train.grad_clip", 1.0))
    checkpoint_path = str(get_by_path(cfg, "output.checkpoint"))
    output_dir = Path(str(get_by_path(cfg, "output.dir", "outputs/experiments/train")))
    patience = int(get_by_path(cfg, "train.early_stopping_patience", 5))

    best_metric = -1.0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        projector.train()
        ranker.train()
        if pairwise is not None:
            pairwise.train()
        running_loss = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            text_emb = batch["text_emb"].to(device)
            frame_emb = batch["frame_emb"].to(device)
            quality = batch["quality"].to(device)
            target = batch["target_perm_idx"].to(device)
            pair_labels = batch["pairwise_labels"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                frame_tokens = projector(text_emb, frame_emb, quality)
                perm_scores = ranker(frame_tokens)
                pair_logits = pairwise(frame_tokens) if pairwise is not None else None
                loss, _loss_metrics = loss_fn(perm_scores, target, pair_logits, pair_labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size_actual = int(text_emb.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size_actual
            seen += batch_size_actual

        result = evaluate_cache(valid_cache, projector, ranker, device, batch_size=batch_size)
        metrics = {k: float(v) for k, v in result["metrics"].items()}
        metrics["train_loss"] = running_loss / max(seen, 1)
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} exact_match={metrics['exact_match']:.4f}")

        if metrics["exact_match"] > best_metric:
            best_metric = metrics["exact_match"]
            best_metrics = metrics
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                projector,
                ranker,
                pairwise,
                cfg,
                embedding_dim,
                quality_dim,
                best_metrics,
            )
            ids = [str(v) for v in valid_cache["ids"].tolist()]
            true_answers = valid_cache["answer"].astype(int).tolist()
            write_eval_outputs(output_dir, ids, result["pred_answers"], true_answers, result["metrics"])
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break

    write_json(output_dir / "metrics.json", {"best": best_metrics, "history": history})
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = train(cfg)
    print(metrics)


if __name__ == "__main__":
    main()

