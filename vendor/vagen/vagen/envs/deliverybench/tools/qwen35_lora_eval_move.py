"""Offline MOVE-direction evaluation for Qwen3.5-VL DeliveryBench LoRA checkpoints.

This script is intentionally separate from SFT training. It can run on a
different GPU, load each saved PEFT adapter checkpoint, generate MOVE actions on
the eval parquet, and write direction-accuracy summaries without blocking the
training loop.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .qwen35_lora_sft_smoke import run_offline_move_eval
from .qwen35_lora_sft_train import DEFAULT_EVAL_PARQUET, DEFAULT_MODEL_PATH, _parse_bool
from .qwen35_visual_sft_dataset import DeliveryBenchQwen35VisualSFTDataset


MOVE_DIRECTIONS = ("forward", "backward", "left", "right")
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _require_peft() -> Any:
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'peft'. Install it in qwen_env before running checkpoint eval."
        ) from exc
    return PeftModel


def _configure_backends(args: argparse.Namespace) -> Dict[str, Any]:
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is not None:
        if hasattr(cuda_backends, "enable_cudnn_sdp"):
            cuda_backends.enable_cudnn_sdp(False)
        if hasattr(cuda_backends, "enable_flash_sdp"):
            cuda_backends.enable_flash_sdp(True)
        if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
            cuda_backends.enable_mem_efficient_sdp(True)
        if hasattr(cuda_backends, "enable_math_sdp"):
            cuda_backends.enable_math_sdp(not bool(args.disable_math_sdp))

    status: Dict[str, Any] = {"cudnn_enabled": bool(torch.backends.cudnn.enabled)}
    for key in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(cuda_backends, key, None) if cuda_backends is not None else None
        if callable(fn):
            status[key] = bool(fn())
    return status


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _checkpoint_ready(path: Path, *, stable_seconds: float) -> bool:
    adapter = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    if not adapter.exists() or not config.exists():
        return False
    if stable_seconds <= 0:
        return True
    newest_mtime = max(adapter.stat().st_mtime, config.stat().st_mtime)
    return (time.time() - newest_mtime) >= float(stable_seconds)


def _summarize_directions(summary: Dict[str, Any]) -> Dict[str, Any]:
    oracle_counts = {direction: 0 for direction in MOVE_DIRECTIONS}
    pred_counts = {direction: 0 for direction in MOVE_DIRECTIONS}
    correct_counts = {direction: 0 for direction in MOVE_DIRECTIONS}
    parsed_counts = {direction: 0 for direction in MOVE_DIRECTIONS}

    for row in summary.get("results", []):
        oracle = row.get("oracle_direction")
        pred = row.get("parsed_predicted_direction")
        if oracle in oracle_counts:
            oracle_counts[oracle] += 1
            if pred is not None:
                parsed_counts[oracle] += 1
            if row.get("matches_oracle"):
                correct_counts[oracle] += 1
        if pred in pred_counts:
            pred_counts[pred] += 1

    accuracy_by_direction = {}
    parse_rate_by_direction = {}
    for direction in MOVE_DIRECTIONS:
        total = oracle_counts[direction]
        accuracy_by_direction[direction] = float(correct_counts[direction] / total) if total else 0.0
        parse_rate_by_direction[direction] = float(parsed_counts[direction] / total) if total else 0.0

    return {
        "oracle_direction_counts": oracle_counts,
        "predicted_direction_counts": pred_counts,
        "correct_by_direction": correct_counts,
        "accuracy_by_direction": accuracy_by_direction,
        "parse_rate_by_direction": parse_rate_by_direction,
    }


def _processor_path(checkpoint_dir: Path, model_path: Path) -> Path:
    if (checkpoint_dir / "processor_config.json").exists() or (checkpoint_dir / "tokenizer_config.json").exists():
        return checkpoint_dir
    return model_path


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> Dict[str, Any]:
    PeftModel = _require_peft()
    checkpoint_dir = checkpoint_dir.resolve()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output) if args.output else checkpoint_dir / "eval_move_acc.json"
    raw_output_path = output_path.with_name(output_path.stem + "_raw.json")

    backend_status = _configure_backends(args)
    processor = AutoProcessor.from_pretrained(
        str(_processor_path(checkpoint_dir, model_path)),
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = DeliveryBenchQwen35VisualSFTDataset(
        args.eval_parquet,
        processor=processor,
        max_samples=args.max_eval_samples,
        stage_filter=args.stage,
        max_length=args.max_length,
        assistant_format="full_json",
        image_policy=args.image_policy,
    )
    if len(dataset) == 0:
        raise RuntimeError("eval dataset is empty")

    device = torch.device(args.device)
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir), is_trainable=False)
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    summary = run_offline_move_eval(
        model=model,
        processor=processor,
        dataset=dataset,
        device=device,
        output_path=raw_output_path,
        sample_limit=args.sample_limit,
        close_thinking=args.close_thinking_for_generation,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
    )
    direction_summary = _summarize_directions(summary)
    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "global_step": _checkpoint_step(checkpoint_dir),
        "model_path": str(model_path),
        "eval_parquet": str(Path(args.eval_parquet).resolve()),
        "sample_limit": int(args.sample_limit),
        "num_samples": int(summary.get("num_samples", 0)),
        "correct": int(summary.get("correct", 0)),
        "accuracy": float(summary.get("accuracy", 0.0)),
        "parsed": int(summary.get("parsed", 0)),
        "parse_rate": float(summary.get("parse_rate", 0.0)),
        "backend_status": backend_status,
        "raw_output_path": str(raw_output_path),
        **direction_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _load_existing_curve(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row.get("checkpoint_dir", ""))] = row
    return rows


def watch_checkpoints(args: argparse.Namespace) -> None:
    watch_dir = Path(args.watch).resolve()
    curve_jsonl = watch_dir / "eval_curve.jsonl"
    curve_json = watch_dir / "eval_curve.json"
    seen = _load_existing_curve(curve_jsonl)
    while True:
        checkpoints = sorted(
            [path for path in watch_dir.glob("checkpoint-*") if path.is_dir()],
            key=_checkpoint_step,
        )
        for checkpoint in checkpoints:
            checkpoint_key = str(checkpoint.resolve())
            if checkpoint_key in seen:
                continue
            if not _checkpoint_ready(checkpoint, stable_seconds=args.stable_seconds):
                continue
            result = evaluate_checkpoint(args, checkpoint)
            seen[checkpoint_key] = result
            with curve_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            curve_json.write_text(
                json.dumps(list(seen.values()), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if args.watch_once:
            return
        time.sleep(float(args.poll_seconds))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default=None, help="PEFT checkpoint directory, e.g. checkpoint-30")
    source.add_argument("--watch", default=None, help="Training output directory containing checkpoint-* dirs")

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-parquet", default=DEFAULT_EVAL_PARQUET)
    parser.add_argument("--output", default=None, help="Output JSON path for single-checkpoint eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--disable-cudnn", type=_parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--disable-math-sdp", action="store_true", default=False)

    parser.add_argument("--stage", action="append", default=["move"])
    parser.add_argument("--image-policy", choices=["all", "drop_first", "last_only"], default="all")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--close-thinking-for-generation", action="store_true")

    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stable-seconds", type=float, default=30.0)
    parser.add_argument("--watch-once", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.checkpoint:
        evaluate_checkpoint(args, Path(args.checkpoint))
    else:
        watch_checkpoints(args)


if __name__ == "__main__":
    main()
