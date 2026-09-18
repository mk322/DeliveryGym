"""Tiny Qwen3.5-VL LoRA SFT smoke test for DeliveryBench visual data.

This is a pipeline validator, not a training recipe. It loads a few exported
DeliveryBench visual SFT rows, injects LoRA into a small set of language-model
linear layers, runs one or two optimization steps, saves the adapter, reloads
it, and performs a basic inference smoke.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from .qwen35_visual_sft_dataset import (
    DeliveryBenchQwen35VisualSFTDataset,
    apply_image_policy,
    as_plain_list,
    collate_qwen35_visual_sft,
    load_pil_images,
    to_qwen_structured_messages,
)


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for a frozen ``nn.Linear`` module."""

    def __init__(self, base: nn.Linear, *, rank: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a.to(device=device, dtype=dtype)
        self.lora_b.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    target_regex: str,
    rank: int,
    alpha: int,
) -> List[str]:
    """Replace matching Linear modules with ``LoRALinear`` wrappers."""
    pattern = re.compile(target_regex)
    matched: List[str] = []
    for name, module in list(model.named_modules()):
        if not pattern.search(name):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, child = _get_parent_module(model, name)
        setattr(parent, child, LoRALinear(module, rank=rank, alpha=alpha))
        matched.append(name)
    if not matched:
        raise RuntimeError(f"no nn.Linear modules matched target_regex={target_regex!r}")
    return matched


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    }


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    missing: List[str] = []
    by_name = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in by_name:
            missing.append(name)
            continue
        by_name[name].data.copy_(tensor.to(device=by_name[name].device, dtype=by_name[name].dtype))
    if missing:
        raise RuntimeError(f"missing LoRA parameters while loading adapter: {missing[:5]}")


def trainable_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for name, param in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            param.requires_grad_(True)
            params.append(param)
        else:
            param.requires_grad_(False)
    return params


def adapter_delta_norm(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, before_tensor in before.items():
        diff = after[name].float() - before_tensor.float()
        total += float(diff.pow(2).sum().item())
    return math.sqrt(total)


def save_adapter(path: Path, *, state: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "adapter.pt")
    (path / "adapter_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(model_path: Path, *, device: torch.device) -> nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.config.use_cache = False
    return model


_MOVE_ACTION_RE = re.compile(r'^MOVE\(direction=["\'](forward|left|right|backward)["\']\)$')


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a generated JSON object, tolerating surrounding chat text."""
    stripped = str(text or "").strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(stripped[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"generated JSON is not an object: {obj!r}")
    return obj


def _validate_generated_move(text: str) -> Dict[str, Any]:
    direction, obj, error = _predicted_move_direction(text)
    if not direction or obj is None:
        raise ValueError(error or f"generated text is not a valid MOVE action: {text!r}")
    return obj


def _move_direction_from_action(action: str) -> str | None:
    match = _MOVE_ACTION_RE.match(str(action or ""))
    if not match:
        return None
    return match.group(1)


def _predicted_move_direction(text: str) -> Tuple[str | None, Dict[str, Any] | None, str | None]:
    stripped = str(text or "").strip()
    direct = _move_direction_from_action(stripped)
    if direct:
        return direct, {"action": stripped}, None
    try:
        obj = _extract_json_object(stripped)
    except Exception as exc:  # noqa: BLE001 - diagnostic string goes into eval JSON.
        return None, None, f"{type(exc).__name__}: {exc}"
    direction = _move_direction_from_action(str(obj.get("action", "")))
    if not direction:
        return None, obj, f"invalid generated action: {obj.get('action')!r}"
    return direction, obj, None


def _generation_batch_from_row(
    dataset: DeliveryBenchQwen35VisualSFTDataset,
    *,
    index: int,
    device: torch.device,
    close_thinking: bool,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Build pre-action system+user tensors for generation.

    The SFT dataset item contains the assistant label in ``input_ids`` so that
    loss computation can supervise it. Generation must instead start from the
    policy-visible prefix only; otherwise the model is asked to continue after a
    completed assistant message and may correctly emit no new tokens.
    """
    row = dataset.dataframe.iloc[int(index)]
    messages = as_plain_list(row["messages"])
    image_entries = as_plain_list(row["images"])
    messages, image_entries = apply_image_policy(
        messages,
        image_entries,
        image_policy=dataset.image_policy,
    )
    qwen_messages = to_qwen_structured_messages(messages, image_entries)
    pil_images = load_pil_images(image_entries)
    prompt = dataset.processor.apply_chat_template(
        qwen_messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    if close_thinking:
        prompt = f"{prompt}\n</think>\n\n"
    inputs = dataset.processor(text=[prompt], images=pil_images, return_tensors="pt")
    tensor_inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    return tensor_inputs, int(tensor_inputs["input_ids"].shape[-1])


def _unique_generation_targets(model: nn.Module) -> List[nn.Module]:
    targets: List[nn.Module] = []

    def add(target: Any) -> None:
        if isinstance(target, nn.Module) and not any(target is existing for existing in targets):
            targets.append(target)

    add(model)
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        add(base_model)
        add(getattr(base_model, "model", None))
    add(getattr(model, "base_model", None))
    add(getattr(model, "model", None))
    return targets


def _trl_original_forward_from_bound_method(forward: Any) -> Any | None:
    func = getattr(forward, "__func__", forward)
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None)
    if code is None or closure is None:
        return None
    for name, cell in zip(code.co_freevars, closure):
        if name != "original_forward":
            continue
        try:
            original = cell.cell_contents
        except ValueError:
            return None
        if callable(original):
            return original
    return None


@contextmanager
def _prepare_qwen35_vl_generation_under_trl_peft(model: nn.Module):
    """Let Qwen3.5-VL generation keep multimodal kwargs under TRL/PEFT.

    TRL's SFTTrainer wraps ``forward`` for chunked CE, so Transformers generation
    validation may no longer see Qwen's explicit ``mm_token_type_ids`` argument.
    The wrapper can also disturb Qwen's multimodal generation path. During
    offline generation eval only, restore the original forward and relax the
    validator for this required field.
    """
    targets = _unique_generation_targets(model)
    originals: List[Tuple[nn.Module, Any]] = []
    forward_originals: List[Tuple[nn.Module, Any]] = []
    for target in targets:
        forward = getattr(target, "forward", None)
        original_forward = _trl_original_forward_from_bound_method(forward)
        if original_forward is not None:
            forward_originals.append((target, forward))
            setattr(target, "forward", original_forward)

        original = getattr(target, "_validate_model_kwargs", None)
        if not callable(original):
            continue

        def relaxed_validate(model_kwargs, *, _original=original):
            filtered = dict(model_kwargs)
            filtered.pop("mm_token_type_ids", None)
            return _original(filtered)

        originals.append((target, original))
        setattr(target, "_validate_model_kwargs", relaxed_validate)
    try:
        yield
    finally:
        for target, original in originals:
            setattr(target, "_validate_model_kwargs", original)
        for target, forward in forward_originals:
            setattr(target, "forward", forward)


def _select_balanced_move_indices(dataset: DeliveryBenchQwen35VisualSFTDataset, limit: int) -> List[int]:
    """Pick a small direction-balanced eval subset in dataframe order."""
    if limit <= 0:
        return []
    by_direction: Dict[str, List[int]] = defaultdict(list)
    for idx, row in dataset.dataframe.iterrows():
        direction = _move_direction_from_action(str(row["action"]))
        if direction:
            by_direction[direction].append(int(idx))
    order = ("backward", "forward", "left", "right")
    selected: List[int] = []
    round_idx = 0
    while len(selected) < limit:
        progressed = False
        for direction in order:
            choices = by_direction.get(direction, [])
            if round_idx < len(choices):
                selected.append(choices[round_idx])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        round_idx += 1
    return selected


def _image_paths_from_row(row: Any, *, image_policy: str = "all") -> List[str]:
    paths: List[str] = []
    _messages, images = apply_image_policy(
        as_plain_list(row["messages"]),
        as_plain_list(row["images"]),
        image_policy=image_policy,
    )
    for entry in images:
        if isinstance(entry, dict):
            paths.append(str(entry.get("image", "")))
        else:
            paths.append(str(entry))
    return paths


def run_offline_move_eval(
    *,
    model: nn.Module,
    processor: Any,
    dataset: DeliveryBenchQwen35VisualSFTDataset,
    device: torch.device,
    output_path: Path,
    sample_limit: int,
    close_thinking: bool,
    max_new_tokens: int,
    min_new_tokens: int,
) -> Dict[str, Any]:
    """Generate MOVE predictions for a small val subset and compute direction accuracy."""
    indices = _select_balanced_move_indices(dataset, int(sample_limit))
    results: List[Dict[str, Any]] = []
    model.eval()
    for idx in indices:
        row = dataset.dataframe.iloc[int(idx)]
        generation_tensors, prompt_len = _generation_batch_from_row(
            dataset,
            index=int(idx),
            device=device,
            close_thinking=close_thinking,
        )
        with torch.no_grad(), _prepare_qwen35_vl_generation_under_trl_peft(model):
            generated = model.generate(
                **generation_tensors,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if int(generated.shape[-1]) > prompt_len:
            generated_new_tokens = generated[:, prompt_len:]
        else:
            generated_new_tokens = generated
        generated_text = processor.batch_decode(generated_new_tokens, skip_special_tokens=True)[0]
        generated_raw_text = processor.batch_decode(generated_new_tokens, skip_special_tokens=False)[0]
        pred_direction, parsed_json, parse_error = _predicted_move_direction(generated_text)
        oracle_direction = _move_direction_from_action(str(row["action"]))
        results.append(
            {
                "row_index": int(idx),
                "seed": int(row["seed"]),
                "turn_index": int(row["turn_index"]),
                "oracle_action": str(row["action"]),
                "oracle_direction": oracle_direction,
                "generated_output": generated_text,
                "generated_raw_output": generated_raw_text,
                "parsed_json": parsed_json,
                "parsed_predicted_direction": pred_direction,
                "parse_error": parse_error,
                "matches_oracle": bool(pred_direction == oracle_direction),
                "image_paths": _image_paths_from_row(row, image_policy=dataset.image_policy),
                "generated_shape": tuple(generated.shape),
                "prompt_len": prompt_len,
                "new_tokens_shape": tuple(generated_new_tokens.shape),
            }
        )
    correct = sum(1 for row in results if row["matches_oracle"])
    parsed = sum(1 for row in results if row["parsed_predicted_direction"] is not None)
    summary = {
        "num_samples": len(results),
        "correct": correct,
        "accuracy": float(correct / len(results)) if results else 0.0,
        "parsed": parsed,
        "parse_rate": float(parsed / len(results)) if results else 0.0,
        "selected_indices": indices,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    train_dataset = DeliveryBenchQwen35VisualSFTDataset(
        args.parquet,
        processor=processor,
        max_samples=args.max_samples,
        stage_filter=args.stage,
        max_length=args.max_length,
        assistant_format=args.assistant_format,
        image_policy=args.image_policy,
    )
    val_dataset = DeliveryBenchQwen35VisualSFTDataset(
        args.val_parquet or args.parquet,
        processor=processor,
        max_samples=args.val_max_samples,
        stage_filter=args.stage,
        max_length=args.max_length,
        assistant_format=args.assistant_format,
        image_policy=args.image_policy,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate_qwen35_visual_sft(samples, pad_token_id=train_dataset.pad_token_id),
    )

    model = load_model(Path(args.model_path), device=device)
    matched = inject_lora(model, target_regex=args.target_regex, rank=args.rank, alpha=args.alpha)
    trainable = trainable_lora_parameters(model)
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    before = lora_state_dict(model)
    losses: List[float] = []
    grad_norms: List[float] = []

    model.train()
    while len(losses) < int(args.steps):
        batches_ran = 0
        for batch in loader:
            if len(losses) >= int(args.steps):
                break
            batches_ran += 1
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            if args.logits_to_keep and "labels" in tensor_batch:
                tensor_batch["labels"] = tensor_batch["labels"][:, -int(args.logits_to_keep) :]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**tensor_batch, logits_to_keep=args.logits_to_keep)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("model did not return loss")
            loss.backward()
            grad_sq = 0.0
            for param in trainable:
                if param.grad is not None:
                    grad_sq += float(param.grad.detach().float().pow(2).sum().item())
            grad_norm = math.sqrt(grad_sq)
            optimizer.step()
            losses.append(float(loss.detach().float().cpu().item()))
            grad_norms.append(grad_norm)
        if batches_ran == 0:
            raise RuntimeError("training dataloader produced no batches")

    after = lora_state_dict(model)
    delta_norm = adapter_delta_norm(before, after)
    if not losses:
        raise RuntimeError("no training steps were run")
    if delta_norm <= 0:
        raise RuntimeError("LoRA parameters did not change")

    metadata = {
        "model_path": str(args.model_path),
        "parquet": [str(p) for p in args.parquet],
        "val_parquet": [str(p) for p in (args.val_parquet or args.parquet)],
        "target_regex": args.target_regex,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "assistant_format": args.assistant_format,
        "image_policy": args.image_policy,
        "steps": len(losses),
        "losses": losses,
        "grad_norms": grad_norms,
        "matched_modules": matched,
        "adapter_delta_norm": delta_norm,
    }
    save_adapter(out_dir, state=after, metadata=metadata)

    # Checkpoint load smoke: zero the adapter, reload saved tensors, then run a
    # tiny greedy generation on the first sample.
    zero_state = {name: torch.zeros_like(tensor) for name, tensor in after.items()}
    load_lora_state_dict(model, zero_state)
    load_lora_state_dict(model, torch.load(out_dir / "adapter.pt", map_location="cpu"))

    model.eval()
    eval_sample = val_dataset[0]
    eval_batch = collate_qwen35_visual_sft([eval_sample], pad_token_id=val_dataset.pad_token_id)
    eval_tensors = {
        key: value.to(device)
        for key, value in eval_batch.items()
        if isinstance(value, torch.Tensor) and key != "labels"
    }
    eval_labels = eval_batch["labels"].to(device)
    if args.logits_to_keep:
        eval_labels = eval_labels[:, -int(args.logits_to_keep) :]
    generation_dataset = train_dataset if args.generation_source == "train" else val_dataset
    generation_tensors, prompt_len = _generation_batch_from_row(
        generation_dataset,
        index=0,
        device=device,
        close_thinking=args.close_thinking_for_generation,
    )
    with torch.no_grad():
        eval_outputs = model(
            **{**eval_tensors, "labels": eval_labels},
            logits_to_keep=args.logits_to_keep,
        )
        generated = model.generate(
            **generation_tensors,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if int(generated.shape[-1]) > prompt_len:
        generated_new_tokens = generated[:, prompt_len:]
    else:
        generated_new_tokens = generated
    generated_text = processor.batch_decode(generated_new_tokens, skip_special_tokens=True)[0]
    generated_raw_text = processor.batch_decode(generated_new_tokens, skip_special_tokens=False)[0]
    generation_debug = {
        "generation_source": args.generation_source,
        "generation_prompt_len": prompt_len,
        "generated_shape": tuple(generated.shape),
        "generated_new_tokens_shape": tuple(generated_new_tokens.shape),
        "generated_token_ids": generated_new_tokens[0].detach().cpu().tolist(),
        "generated_text": generated_text,
        "generated_raw_text": generated_raw_text,
        "close_thinking_for_generation": args.close_thinking_for_generation,
    }
    (out_dir / "generation_debug.json").write_text(
        json.dumps(generation_debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    generated_json = _validate_generated_move(generated_text)
    offline_eval = None
    if int(args.eval_val_samples) > 0:
        eval_dataset = DeliveryBenchQwen35VisualSFTDataset(
            args.val_parquet or args.parquet,
            processor=processor,
            max_samples=-1,
            stage_filter=args.stage,
            max_length=args.max_length,
            assistant_format=args.assistant_format,
            image_policy=args.image_policy,
        )
        offline_eval = run_offline_move_eval(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            device=device,
            output_path=out_dir / "offline_eval.json",
            sample_limit=int(args.eval_val_samples),
            close_thinking=args.close_thinking_for_generation,
            max_new_tokens=int(args.eval_max_new_tokens),
            min_new_tokens=int(args.min_new_tokens),
        )
    metadata.update(
        {
            "val_loss": float(eval_outputs.loss.detach().float().cpu().item()),
            "generation_source": args.generation_source,
            "close_thinking_for_generation": args.close_thinking_for_generation,
            "generated_shape": tuple(generated.shape),
            "generation_prompt_len": prompt_len,
            "generated_new_tokens_shape": tuple(generated_new_tokens.shape),
            "generated_text": generated_text,
            "generated_json": generated_json,
            "generated_action": str(generated_json["action"]),
            "generated_valid_json_move": True,
            "checkpoint_dir": str(out_dir),
            "trainable_lora_params": int(sum(param.numel() for param in trainable)),
        }
    )
    if offline_eval is not None:
        metadata["offline_eval"] = {
            key: value
            for key, value in offline_eval.items()
            if key != "results"
        }
    (out_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--val-parquet", action="append", default=None)
    parser.add_argument("--output-dir", default="/tmp/deliverybench_qwen35_lora_sft_smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--val-max-samples", type=int, default=2)
    parser.add_argument("--assistant-format", choices=["full_json", "action_json", "action"], default="full_json")
    parser.add_argument("--image-policy", choices=["all", "drop_first", "last_only"], default="all")
    parser.add_argument("--stage", action="append", default=["move"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--target-regex",
        default=r"model\.language_model\.layers\.31\.self_attn\.(q_proj|v_proj)$",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--generation-source", choices=["train", "val"], default="val")
    parser.add_argument("--close-thinking-for-generation", action="store_true")
    parser.add_argument("--eval-val-samples", type=int, default=0)
    parser.add_argument("--logits-to-keep", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_smoke(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
