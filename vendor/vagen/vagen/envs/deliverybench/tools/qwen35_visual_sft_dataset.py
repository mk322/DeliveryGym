"""Qwen3.5-VL dataset adapter for exported DeliveryBench visual SFT parquet.

This module does not train a model. It only converts the exported
``messages``/``images`` parquet schema into Qwen structured multimodal chat
content, runs the Qwen processor, and returns tensors suitable for a later
LoRA SFT step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_PLACEHOLDER = "<image>"


def as_plain_list(value: Any) -> List[Any]:
    """Convert pandas/pyarrow/numpy nested values back to regular lists."""
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _image_path(image_entry: Any) -> str:
    if isinstance(image_entry, dict):
        path = image_entry.get("image")
    else:
        path = image_entry
    if not path:
        raise ValueError(f"invalid image entry: {image_entry!r}")
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return path


def load_pil_images(images: Sequence[Any]) -> List[Image.Image]:
    """Load image entries from the exported parquet as RGB PIL images."""
    pil_images: List[Image.Image] = []
    for image in images:
        pil_images.append(Image.open(_image_path(image)).convert("RGB"))
    return pil_images


def to_qwen_structured_messages(
    messages: Sequence[Dict[str, Any]],
    images: Sequence[Any],
    *,
    image_placeholder: str = IMAGE_PLACEHOLDER,
) -> List[Dict[str, Any]]:
    """Replace ``<image>`` placeholders with Qwen structured image parts.

    Exported DeliveryBench rows store images separately and keep placeholders in
    the user text. Qwen3.5-VL requires structured message content such as
    ``{"type": "image", "image": "/path/to.png"}`` before applying the chat
    template. This function performs that conversion while preserving text.
    """
    image_entries = list(images)
    image_offset = 0
    converted: List[Dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if not isinstance(content, str) or image_placeholder not in content:
            converted.append({"role": role, "content": content})
            continue

        parts: List[Dict[str, str]] = []
        for segment in re.split(f"({re.escape(image_placeholder)})", content):
            if not segment:
                continue
            if segment == image_placeholder:
                if image_offset >= len(image_entries):
                    raise AssertionError(
                        f"more {image_placeholder} placeholders than images: "
                        f"offset={image_offset}, images={len(image_entries)}"
                    )
                parts.append({"type": "image", "image": _image_path(image_entries[image_offset])})
                image_offset += 1
            else:
                parts.append({"type": "text", "text": segment})
        converted.append({"role": role, "content": parts})

    if image_offset != len(image_entries):
        raise AssertionError(f"image placeholders used {image_offset}, but row has {len(image_entries)} images")
    return converted


def _messages_and_images_from_row(row: Any) -> tuple[List[Dict[str, Any]], List[Any]]:
    messages = as_plain_list(row["messages"])
    images = as_plain_list(row["images"])
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise AssertionError(f"expected system/user/assistant messages, got {[m.get('role') for m in messages]}")
    return messages, images


def _select_images(images: Sequence[Any], image_policy: str) -> List[Any]:
    """Select image inputs for training without modifying the source parquet."""
    images = list(images)
    if image_policy == "all":
        return images
    if image_policy == "drop_first":
        return images[1:] if len(images) > 1 else images
    if image_policy == "last_only":
        return images[-1:] if images else []
    raise ValueError(f"unknown image_policy={image_policy!r}")


def _rewrite_image_placeholders(
    messages: Sequence[Dict[str, Any]],
    *,
    num_images: int,
    image_placeholder: str = IMAGE_PLACEHOLDER,
) -> List[Dict[str, Any]]:
    """Move selected image placeholders to the start of the first user message.

    DeliveryBench observations store image placeholders as a prefix
    (``<image><image>\n\n...``). When a training run ignores FPV images, the
    parquet row should stay unchanged, but the model-facing user message must
    have the same number of placeholders as the selected images.
    """
    converted: List[Dict[str, Any]] = []
    rewrote_user = False
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if role == "user" and isinstance(content, str) and not rewrote_user:
            text = content.replace(image_placeholder, "").lstrip()
            prefix = image_placeholder * int(num_images)
            content = f"{prefix}\n\n{text}" if prefix else text
            rewrote_user = True
        converted.append({"role": role, "content": content})
    return converted


def apply_image_policy(
    messages: Sequence[Dict[str, Any]],
    images: Sequence[Any],
    *,
    image_policy: str,
    image_placeholder: str = IMAGE_PLACEHOLDER,
) -> tuple[List[Dict[str, Any]], List[Any]]:
    """Return model-facing messages/images after optional FPV filtering."""
    selected_images = _select_images(images, image_policy)
    if image_policy == "all":
        return [dict(message) for message in messages], selected_images
    selected_messages = _rewrite_image_placeholders(
        messages,
        num_images=len(selected_images),
        image_placeholder=image_placeholder,
    )
    return selected_messages, selected_images


def _format_assistant_message(content: str, assistant_format: str) -> str:
    """Optionally simplify assistant targets for small SFT sanity runs."""
    if assistant_format == "full_json":
        return content
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant content is not JSON: {content[:200]!r}") from exc
    action = str(obj.get("action", ""))
    if not action:
        raise ValueError(f"assistant JSON has no action: {content[:200]!r}")
    if assistant_format == "action_json":
        return json.dumps({"action": action}, ensure_ascii=False)
    if assistant_format == "action":
        return action
    raise ValueError(f"unknown assistant_format={assistant_format!r}")


class DeliveryBenchQwen35VisualSFTDataset(Dataset):
    """Minimal Qwen3.5-VL SFT dataset for exported DeliveryBench visual rows."""

    def __init__(
        self,
        parquet_files: str | Path | Sequence[str | Path],
        *,
        processor: Any,
        max_samples: int = -1,
        stage_filter: Optional[Iterable[str]] = None,
        max_length: Optional[int] = None,
        assistant_format: str = "full_json",
        image_policy: str = "all",
    ) -> None:
        if isinstance(parquet_files, (str, Path)):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(Path(path)) for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if stage_filter is not None:
            stages = {str(stage) for stage in stage_filter}
            self.dataframe = self.dataframe[self.dataframe["stage"].astype(str).isin(stages)].reset_index(drop=True)
        if max_samples is not None and int(max_samples) > 0:
            self.dataframe = self.dataframe.head(int(max_samples)).reset_index(drop=True)

        self.processor = processor
        self.max_length = int(max_length) if max_length is not None else None
        self.assistant_format = str(assistant_format)
        self.image_policy = str(image_policy)
        self.pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[int(index)]
        messages, image_entries = _messages_and_images_from_row(row)
        messages, image_entries = apply_image_policy(
            messages,
            image_entries,
            image_policy=self.image_policy,
        )
        messages[-1]["content"] = _format_assistant_message(
            str(messages[-1].get("content", "")),
            self.assistant_format,
        )
        qwen_messages = to_qwen_structured_messages(messages, image_entries)
        pil_images = load_pil_images(image_entries)

        full_prompt = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_prompt = self.processor.apply_chat_template(
            qwen_messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_inputs = self.processor(text=[full_prompt], images=pil_images, return_tensors="pt")
        prefix_inputs = self.processor(text=[prefix_prompt], images=pil_images, return_tensors="pt")

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()
        prefix_len = int(prefix_inputs["input_ids"].shape[-1])
        labels[:prefix_len] = -100

        if self.max_length is not None and input_ids.shape[-1] > self.max_length:
            raise ValueError(f"sample length {input_ids.shape[-1]} exceeds max_length={self.max_length}")

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": {
                "seed": int(row["seed"]),
                "turn_index": int(row["turn_index"]),
                "stage": str(row["stage"]),
                "action": str(row["action"]),
                "num_images": len(pil_images),
                "image_policy": self.image_policy,
                "prefix_len": prefix_len,
                "sequence_len": int(input_ids.shape[-1]),
            },
        }
        for key, value in full_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            out[key] = value
        return out


def collate_qwen35_visual_sft(samples: Sequence[Dict[str, Any]], *, pad_token_id: int = 0) -> Dict[str, Any]:
    """Pad text tensors and concatenate Qwen visual tensors for a small batch."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_len = max(int(sample["input_ids"].shape[-1]) for sample in samples)

    input_ids, attention_mask, labels, mm_token_type_ids = [], [], [], []
    for sample in samples:
        seq_len = int(sample["input_ids"].shape[-1])
        pad_len = max_len - seq_len
        input_ids.append(torch.nn.functional.pad(sample["input_ids"], (0, pad_len), value=int(pad_token_id)))
        attention_mask.append(torch.nn.functional.pad(sample["attention_mask"], (0, pad_len), value=0))
        labels.append(torch.nn.functional.pad(sample["labels"], (0, pad_len), value=-100))
        if "mm_token_type_ids" in sample:
            mm_ids = sample["mm_token_type_ids"]
            if mm_ids.dim() == 2:
                mm_ids = mm_ids[0]
            mm_token_type_ids.append(torch.nn.functional.pad(mm_ids, (0, pad_len), value=0))

    batch: Dict[str, Any] = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "labels": torch.stack(labels, dim=0),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }
    if mm_token_type_ids:
        batch["mm_token_type_ids"] = torch.stack(mm_token_type_ids, dim=0)

    for key in ("pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        values = [sample[key] for sample in samples if key in sample]
        if values:
            batch[key] = torch.cat(values, dim=0)
    return batch


def summarize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable shape summary for smoke tests/debugging."""
    summary = {
        "input_ids_shape": tuple(sample["input_ids"].shape),
        "attention_mask_shape": tuple(sample["attention_mask"].shape),
        "labels_shape": tuple(sample["labels"].shape),
        "labels_supervised": int((sample["labels"] != -100).sum().item()),
        "metadata": dict(sample.get("metadata", {})),
    }
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids", "video_grid_thw", "second_per_grid_ts"):
        if key in sample:
            summary[f"{key}_shape"] = tuple(sample[key].shape)
    return summary


def _load_processor(model_path: str) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test DeliveryBench visual SFT parquet with Qwen3.5-VL processor.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parquet", required=True, action="append")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--stage", action="append", default=None, help="optional stage filter, e.g. --stage move")
    parser.add_argument("--image-policy", choices=["all", "drop_first", "last_only"], default="all")
    args = parser.parse_args()

    processor = _load_processor(args.model_path)
    dataset = DeliveryBenchQwen35VisualSFTDataset(
        args.parquet,
        processor=processor,
        max_samples=args.max_samples,
        stage_filter=args.stage,
        image_policy=args.image_policy,
    )
    rows = []
    for idx in range(len(dataset)):
        rows.append(summarize_sample(dataset[idx]))
    batch = collate_qwen35_visual_sft([dataset[idx] for idx in range(len(dataset))], pad_token_id=dataset.pad_token_id)
    print(json.dumps({"num_samples": len(dataset), "samples": rows, "batch_keys": sorted(batch.keys())}, indent=2))


if __name__ == "__main__":
    main()
