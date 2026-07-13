from snu_order.qwen3vl.compare_experiment_configs import compare_configs


def _base():
    return {
        "experiment": {"id": "e1", "run_id": "e1", "seed": 42},
        "train": {"lora_lr": 5e-5},
        "loss": {"pair_weight": 0.2},
        "processor": {"max_pixels": None},
        "vision_merger_lora": {"enabled": False},
    }


def test_only_allowed_config_paths_succeed():
    candidate = _base()
    candidate["experiment"] = {"id": "e2", "run_id": "e2", "seed": 42}
    candidate["vision_merger_lora"] = {"enabled": True}
    differences = compare_configs(
        _base(),
        candidate,
        allowed_paths={
            "experiment.id",
            "experiment.run_id",
            "vision_merger_lora.enabled",
        },
    )
    assert differences == {}


def test_lr_loss_seed_and_pixels_differences_fail():
    candidate = _base()
    candidate["experiment"]["seed"] = 7
    candidate["train"]["lora_lr"] = 1e-4
    candidate["loss"]["pair_weight"] = 0.5
    candidate["processor"]["max_pixels"] = 1024
    differences = compare_configs(_base(), candidate, allowed_paths=set())
    assert set(differences) == {
        "experiment.seed",
        "train.lora_lr",
        "loss.pair_weight",
        "processor.max_pixels",
    }
