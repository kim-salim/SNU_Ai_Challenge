from __future__ import annotations


def get_device(prefer_cuda: bool = True) -> str:
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

