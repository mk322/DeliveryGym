"""Smoke tests for the Qwen3.5-VL DeliveryBench visual SFT adapter.

Run:
    PYTHONPATH=. python -m vagen.envs.deliverybench.test_qwen35_visual_sft_dataset
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


class _FakeTokenizer:
    pad_token_id = 0


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        chunks = []
        for message in messages:
            chunks.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, str):
                chunks.append(content)
                continue
            for part in content:
                if part["type"] == "image":
                    chunks.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif part["type"] == "text":
                    chunks.append(part["text"])
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "\n".join(chunks)

    def __call__(self, *, text, images, return_tensors):
        assert return_tensors == "pt"
        prompt = text[0]
        image_count = len(images or [])
        length = max(8, len(prompt.split()) + prompt.count("<|image_pad|>") * 5)
        return {
            "input_ids": torch.arange(length, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "pixel_values": torch.ones((image_count * 4, 1536), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * image_count, dtype=torch.long),
        }


def test_adapter_smoke() -> None:
    from vagen.envs.deliverybench.tools.qwen35_visual_sft_dataset import (
        DeliveryBenchQwen35VisualSFTDataset,
        apply_image_policy,
        collate_qwen35_visual_sft,
        summarize_sample,
        to_qwen_structured_messages,
    )

    out_dir = Path("/tmp/deliverybench_qwen35_visual_sft_dataset_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True)
    img0 = image_dir / "img0.png"
    img1 = image_dir / "img1.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(img0)
    Image.new("RGB", (20, 12), (0, 0, 255)).save(img1)

    messages = [
        {"role": "system", "content": "You are a delivery agent."},
        {"role": "user", "content": "<image><image>\n\n### agent_state\nFollow the visual route."},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "reasoning_and_reflection": "Follow the blue route.",
                    "action": 'MOVE(direction="forward")',
                    "future_plan": "continue route following",
                }
            ),
        },
    ]
    images = [{"image": str(img0.resolve())}, {"image": str(img1.resolve())}]
    qwen_messages = to_qwen_structured_messages(messages, images)
    assert qwen_messages[1]["content"][0]["type"] == "image"
    assert qwen_messages[1]["content"][1]["type"] == "image"
    map_messages, map_images = apply_image_policy(messages, images, image_policy="drop_first")
    assert len(map_images) == 1
    assert map_images[0]["image"] == str(img1.resolve())
    assert map_messages[1]["content"].count("<image>") == 1

    parquet = out_dir / "tiny.parquet"
    pd.DataFrame(
        [
            {
                "messages": messages,
                "images": images,
                "seed": 1,
                "turn_index": 2,
                "stage": "move",
                "action": 'MOVE(direction="forward")',
                "repeat_index": 0,
            },
            {
                "messages": messages,
                "images": images,
                "seed": 1,
                "turn_index": 3,
                "stage": "move",
                "action": 'MOVE(direction="forward")',
                "repeat_index": 1,
            },
        ]
    ).to_parquet(parquet, index=False)

    dataset = DeliveryBenchQwen35VisualSFTDataset(parquet, processor=_FakeProcessor(), max_samples=2)
    sample = dataset[0]
    summary = summarize_sample(sample)
    assert summary["metadata"]["stage"] == "move"
    assert summary["metadata"]["num_images"] == 2
    assert summary["labels_supervised"] > 0
    assert sample["labels"][: summary["metadata"]["prefix_len"]].eq(-100).all()
    assert sample["pixel_values"].shape == (8, 1536)
    assert sample["image_grid_thw"].shape == (2, 3)

    map_dataset = DeliveryBenchQwen35VisualSFTDataset(
        parquet,
        processor=_FakeProcessor(),
        max_samples=1,
        image_policy="drop_first",
    )
    map_sample = map_dataset[0]
    map_summary = summarize_sample(map_sample)
    assert map_summary["metadata"]["num_images"] == 1
    assert map_summary["metadata"]["image_policy"] == "drop_first"
    assert map_sample["pixel_values"].shape == (4, 1536)
    assert map_sample["image_grid_thw"].shape == (1, 3)

    batch = collate_qwen35_visual_sft([dataset[0], dataset[1]], pad_token_id=dataset.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["pixel_values"].shape == (16, 1536)
    assert batch["image_grid_thw"].shape == (4, 3)
    print("PASS test_adapter_smoke")


if __name__ == "__main__":
    test_adapter_smoke()
