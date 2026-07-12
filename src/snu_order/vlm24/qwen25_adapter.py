from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from snu_order.vlm24.candidates import candidate_label_to_answer, candidate_label_to_order
from snu_order.vlm24.image_builder import make_labeled_grid_2x2


class Qwen25VLAdapter:
    def __init__(self, config: dict[str, Any], image_mode: str | None = None, scoring_mode: str | None = None):
        self.config = config
        self.model = None
        self.processor = None
        self.image_mode = image_mode or str(config.get("input", {}).get("image_mode", "multi_image"))
        self.scoring_mode = scoring_mode or str(config.get("scoring", {}).get("mode", "option_label_logprob"))

    def _dtype(self, value: str | None) -> torch.dtype | None:
        if value is None:
            return None
        normalized = str(value).lower()
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {value}")

    def _bnb_config(self) -> Any | None:
        quant_cfg = self.config.get("quantization", {})
        if not bool(quant_cfg.get("enabled", False)):
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("4-bit quantization requires transformers BitsAndBytesConfig and bitsandbytes") from exc
        return BitsAndBytesConfig(
            load_in_4bit=bool(quant_cfg.get("load_in_4bit", True)),
            bnb_4bit_compute_dtype=self._dtype(quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        )

    def load_model_and_processor(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        try:
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen VLM inference requires transformers with Qwen VL model support"
            ) from exc

        model_cfg = self.config.get("model", {})
        processor_cfg = self.config.get("processor", {})
        attention_cfg = self.config.get("attention", {})
        model_name = str(model_cfg.get("name", "Qwen/Qwen2.5-VL-7B-Instruct"))
        model_type = str(model_cfg.get("type", "qwen2_5_vl")).lower()
        local_files_only = bool(model_cfg.get("local_files_only", True))
        if model_type == "qwen3_vl":
            model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            if model_class is None:
                model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_class is None:
                raise ImportError(
                    "Qwen3-VL inference requires transformers with Qwen3VLForConditionalGeneration "
                    "or AutoModelForMultimodalLM support"
                )
        elif model_type == "qwen2_5_vl":
            model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            if model_class is None:
                raise ImportError(
                    "Qwen2.5-VL inference requires transformers with Qwen2_5_VLForConditionalGeneration support"
                )
        else:
            raise ValueError(f"Unsupported VLM model type: {model_type}")

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "device_map": model_cfg.get("device_map", "auto"),
        }
        dtype = self._dtype(model_cfg.get("torch_dtype", "bfloat16"))
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        quantization_config = self._bnb_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if bool(attention_cfg.get("use_flash_attention_2", False)):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        processor_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if processor_cfg.get("min_pixels") is not None:
            processor_kwargs["min_pixels"] = int(processor_cfg["min_pixels"])
        if processor_cfg.get("max_pixels") is not None:
            processor_kwargs["max_pixels"] = int(processor_cfg["max_pixels"])

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        try:
            self.model = model_class.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            if model_kwargs.get("attn_implementation") == "flash_attention_2" and bool(
                attention_cfg.get("fallback_to_sdpa", True)
            ):
                warnings.warn(
                    f"flash_attention_2 load failed ({exc}); retrying with sdpa/default attention",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model_kwargs.pop("attn_implementation", None)
                try:
                    model_kwargs["attn_implementation"] = "sdpa"
                    self.model = model_class.from_pretrained(model_name, **model_kwargs)
                except Exception:
                    model_kwargs.pop("attn_implementation", None)
                    self.model = model_class.from_pretrained(model_name, **model_kwargs)
            else:
                raise
        self.model.eval()
        if bool(self.config.get("lora", {}).get("enabled", False)):
            raise NotImplementedError(
                "LoRA hooks are reserved in config, but LoRA loading/training is disabled in the first pass"
            )

    @property
    def tokenizer(self) -> Any:
        self.load_model_and_processor()
        return self.processor.tokenizer

    def _model_device(self) -> torch.device:
        self.load_model_and_processor()
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_messages(self, prompt: str, images: Sequence[Image.Image]) -> list[dict[str, Any]]:
        input_cfg = self.config.get("input", {})
        frame_labels = list(input_cfg.get("frame_labels", ["F1", "F2", "F3", "F4"]))
        mode = self.image_mode
        if mode == "grid_2x2":
            if len(images) == 1:
                image_payload = images[0]
            else:
                image_payload = make_labeled_grid_2x2(
                    [image.convert("RGB") for image in images],
                    labels=frame_labels,
                    grid_size=int(input_cfg.get("grid_size", 1024)),
                    label_height=int(input_cfg.get("grid_label_height", 40)),
                )
            content = [{"type": "image", "image": image_payload}, {"type": "text", "text": prompt}]
        elif mode == "multi_image":
            if len(images) != 4:
                raise ValueError(f"multi_image mode requires 4 images, got {len(images)}")
            content = [{"type": "image", "image": image.convert("RGB")} for image in images]
            content.append({"type": "text", "text": prompt})
        else:
            raise ValueError(f"Unsupported image mode: {mode}")
        return [{"role": "user", "content": content}]

    def build_inputs(self, prompt: str, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        self.load_model_and_processor()
        messages = self.build_messages(prompt, images)
        model_type = str(self.config.get("model", {}).get("type", "qwen2_5_vl")).lower()
        if model_type == "qwen3_vl":
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            return self._move_inputs_to_model_device(inputs)

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("Qwen2.5-VL image preprocessing requires qwen-vl-utils") from exc

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return self._move_inputs_to_model_device(inputs)

    def _move_inputs_to_model_device(self, inputs: Any) -> dict[str, torch.Tensor]:
        device = self._model_device()
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved

    def _variant_texts(self, label: str) -> list[str]:
        variants = self.config.get("scoring", {}).get("label_token_variants", ["{label}", " {label}"])
        return [str(variant).format(label=label) for variant in variants]

    def _score_sequence_logprob(self, inputs: dict[str, torch.Tensor], token_ids: list[int]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        seq = torch.tensor([token_ids], dtype=input_ids.dtype, device=input_ids.device)
        full_inputs = dict(inputs)
        full_inputs["input_ids"] = torch.cat([input_ids, seq], dim=1)
        for key, value in inputs.items():
            if key == "input_ids" or not torch.is_tensor(value):
                continue
            if value.ndim == 2 and value.shape == input_ids.shape:
                if key == "attention_mask":
                    extension = torch.ones_like(seq, dtype=value.dtype, device=value.device)
                else:
                    extension = torch.zeros_like(seq, dtype=value.dtype, device=value.device)
                full_inputs[key] = torch.cat([value, extension], dim=1)
        outputs = self.model(**full_inputs)
        logits = outputs.logits
        prompt_len = input_ids.shape[1]
        total = logits.new_tensor(0.0)
        for offset, token_id in enumerate(token_ids):
            log_probs = logits[0, prompt_len + offset - 1].log_softmax(dim=-1)
            total = total + log_probs[int(token_id)]
        if bool(self.config.get("scoring", {}).get("normalize_sequence_logprob", True)):
            total = total / max(len(token_ids), 1)
        return total

    def score_option_labels(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        labels: Sequence[str],
    ) -> torch.Tensor:
        self.load_model_and_processor()
        inputs = self.build_inputs(prompt, images)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            log_probs = logits.log_softmax(dim=-1)
            label_scores = []
            for label in labels:
                variant_scores = []
                for variant in self._variant_texts(str(label)):
                    token_ids = self.processor.tokenizer.encode(variant, add_special_tokens=False)
                    if not token_ids:
                        continue
                    if len(token_ids) == 1:
                        variant_scores.append(log_probs[0, int(token_ids[0])])
                    else:
                        variant_scores.append(self._score_sequence_logprob(inputs, [int(v) for v in token_ids]))
                if not variant_scores:
                    raise ValueError(f"No token variants could be scored for label {label!r}")
                label_scores.append(torch.stack(variant_scores).max())
            return torch.stack(label_scores).detach().float().cpu()

    def generate_option(self, prompt: str, images: Sequence[Image.Image]) -> tuple[str | None, str]:
        self.load_model_and_processor()
        inputs = self.build_inputs(prompt, images)
        scoring_cfg = self.config.get("scoring", {})
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(scoring_cfg.get("max_new_tokens", 4)),
                do_sample=bool(scoring_cfg.get("do_sample", False)),
                temperature=float(scoring_cfg.get("temperature", 0.0)),
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        raw_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return self.parse_option(raw_text), raw_text

    @staticmethod
    def parse_option(raw_text: str) -> str | None:
        text = str(raw_text).strip()
        if not text:
            return None
        keyword_match = re.search(
            r"(?:answer|option|choice|final)\s*(?:is|:)?\s*([A-X])\b",
            text,
            flags=re.IGNORECASE,
        )
        if keyword_match:
            return keyword_match.group(1).upper()
        direct_match = re.match(r"^\s*([A-X])(?:[\s\.\):,-]|$)", text, flags=re.IGNORECASE)
        if direct_match:
            return direct_match.group(1).upper()
        matches = re.findall(r"\b([A-X])\b", text, flags=re.IGNORECASE)
        if len(matches) == 1:
            return matches[0].upper()
        return None

    def predict_one(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        labels = [str(candidate["label"]) for candidate in candidates]
        mode = self.scoring_mode
        raw_output = ""
        scores: list[float] | None = None
        parse_status = "ok"
        if mode == "option_label_logprob":
            try:
                score_tensor = self.score_option_labels(prompt, images, labels)
                scores = [float(v) for v in score_tensor.tolist()]
                best_idx = int(score_tensor.argmax().item())
                pred_option = labels[best_idx]
            except Exception as exc:
                fallback = str(self.config.get("scoring", {}).get("fallback_mode", "direct_generation"))
                if fallback != "direct_generation":
                    raise
                parse_status = f"logprob_failed_fallback:{type(exc).__name__}"
                pred_option, raw_output = self.generate_option(prompt, images)
        elif mode == "direct_generation":
            pred_option, raw_output = self.generate_option(prompt, images)
        else:
            raise ValueError(f"Unsupported scoring mode: {mode}")

        if pred_option is None:
            return {
                "pred_option": None,
                "pred_order": None,
                "pred_answer": None,
                "scores": scores,
                "raw_output": raw_output,
                "parse_status": "parse_failed" if parse_status == "ok" else parse_status,
            }
        return {
            "pred_option": pred_option,
            "pred_order": list(candidate_label_to_order(pred_option, candidates)),
            "pred_answer": candidate_label_to_answer(pred_option, candidates),
            "scores": scores,
            "raw_output": raw_output,
            "parse_status": parse_status,
        }
