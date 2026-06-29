from __future__ import annotations

from pathlib import Path
from typing import Any

from snu_order.encoders.offline_loader import require_local_model_dir, set_offline_env
from snu_order.utils.device import get_device


class SigLIP2Encoder:
    def __init__(self, model_dir: str | Path, device: str | None = None, dtype: str = "fp16"):
        try:
            import torch
            import torch.nn.functional as F  # noqa: F401
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "SigLIP2Encoder requires torch and transformers. Install requirements.txt first."
            ) from exc

        set_offline_env()
        local_dir = require_local_model_dir(model_dir)
        self.device = device or get_device(prefer_cuda=True)
        self.dtype_name = dtype
        torch_dtype = self._resolve_dtype(dtype, torch)

        self.processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        model_kwargs: dict[str, Any] = {"local_files_only": True}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(local_dir, **model_kwargs).to(self.device)
        if self.device != "cuda" and torch_dtype == torch.float16:
            self.model = self.model.float()
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype: str, torch: Any) -> Any:
        normalized = dtype.lower()
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "full"}:
            return torch.float32
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _move_inputs(self, inputs: Any) -> Any:
        return {key: value.to(self.device) for key, value in inputs.items()}

    def encode_text(self, sentences: list[str]) -> Any:
        import torch
        import torch.nn.functional as F

        if not sentences:
            raise ValueError("sentences must be non-empty")
        inputs = self.processor(text=sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                emb = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                emb = getattr(outputs, "text_embeds", None)
                if emb is None:
                    emb = getattr(outputs, "pooler_output", None)
                if emb is None:
                    raise RuntimeError("Could not find text embeddings in model outputs")
        return F.normalize(emb.float(), dim=-1)

    def encode_images(self, images: list[Any]) -> Any:
        import torch
        import torch.nn.functional as F

        if not images:
            raise ValueError("images must be non-empty")
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                emb = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                emb = getattr(outputs, "image_embeds", None)
                if emb is None:
                    emb = getattr(outputs, "pooler_output", None)
                if emb is None:
                    raise RuntimeError("Could not find image embeddings in model outputs")
        return F.normalize(emb.float(), dim=-1)

    def encode_batch(self, sentences: list[str], frames: list[list[Any]]) -> tuple[Any, Any]:
        if len(sentences) != len(frames):
            raise ValueError(f"sentences and frames length mismatch: {len(sentences)} vs {len(frames)}")
        for sample in frames:
            if len(sample) != 4:
                raise ValueError("Each sample must contain exactly 4 frames")
        text_emb = self.encode_text(sentences)
        flat_images = [frame for sample in frames for frame in sample]
        image_emb = self.encode_images(flat_images)
        frame_emb = image_emb.reshape(len(sentences), 4, -1)
        return text_emb, frame_emb
