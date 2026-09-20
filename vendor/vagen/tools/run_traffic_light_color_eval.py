"""Classify red/green traffic-light color on saved hazard batch images."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from openai import AsyncOpenAI
from PIL import Image


HERE = Path(__file__).resolve().parents[1] / "vagen" / "envs" / "deliverybench"
OUT_ROOT = HERE / "outputs"


def load_case_results(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("event") == "case_result":
                rows.append(rec)
    return rows


def image_data_url(path: Path, max_side: int) -> str:
    im = Image.open(path).convert("RGB")
    scale = min(1.0, float(max_side) / max(im.size))
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_color(text: str) -> Optional[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.S)
        data = {}
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = {}
    value = str(data.get("color", "")).lower() if isinstance(data, dict) else ""
    if value in {"red", "green"}:
        return value
    low = text.lower()
    if "green" in low and "red" not in low:
        return "green"
    if "red" in low and "green" not in low:
        return "red"
    return None


def selected_cases(rows: List[Mapping[str, Any]], label: str, n: int) -> List[Mapping[str, Any]]:
    selected = [r for r in rows if r.get("test_type") == label]
    selected.sort(key=lambda r: str(r.get("case_id", "")))
    return selected[:n]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-image-side", type=int, default=960)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--source-log",
        action="append",
        type=Path,
        required=True,
        help="Previous hazard batch log.jsonl to reuse exact saved image paths.",
    )
    args = parser.parse_args()

    rows = load_case_results(args.source_log)
    cases = selected_cases(rows, "traffic_red", args.n) + selected_cases(rows, "traffic_green", args.n)
    out_dir = OUT_ROOT / f"traffic_light_color_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180)
    system_text = "You are a careful visual classifier for traffic-light color."
    prompt = (
        "Look at the FRONT VIEW traffic signal in the image. "
        "Identify only the currently illuminated traffic-light color. "
        "Return JSON only with keys reasoning and color. "
        "The color value must be exactly red or green."
    )
    results: List[Dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as fh:
        def emit(record: Dict[str, Any]) -> None:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(record)

        emit({
            "event": "run_meta",
            "run_dir": str(out_dir),
            "source_logs": [str(p) for p in args.source_log],
            "model": args.model,
            "n_per_label": args.n,
        })
        for case in cases:
            image_path = Path(case["input"]["image_paths"][0])
            case_id = str(case["case_id"])
            input_dir = out_dir / "inputs" / case_id
            input_dir.mkdir(parents=True, exist_ok=True)
            request = {
                "source_case_id": case_id,
                "source_test_type": case.get("test_type"),
                "source_map_name": case.get("map_name"),
                "source_image_path": str(image_path),
                "system_prompt": system_text,
                "user_text": prompt,
            }
            (input_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            (input_dir / "system_prompt.txt").write_text(system_text, encoding="utf-8")
            (input_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_text},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url(image_path, args.max_image_side)}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=args.temperature,
                top_p=0.9,
                max_tokens=args.max_tokens,
            )
            text = resp.choices[0].message.content or ""
            emit({
                "event": "case_result",
                "case_id": case_id,
                "expected_color": "red" if case.get("test_type") == "traffic_red" else "green",
                "source_test_type": case.get("test_type"),
                "source_map_name": case.get("map_name"),
                "source_image_path": str(image_path),
                "model_response": text,
                "predicted_color": parse_color(text),
                "input": {
                    "request_path": str(input_dir / "request.json"),
                    "system_prompt_path": str(input_dir / "system_prompt.txt"),
                    "user_prompt_path": str(input_dir / "user_prompt.txt"),
                    "image_paths": [str(image_path)],
                },
            })

        rows_out = [r for r in results if r.get("event") == "case_result"]
        summary = {
            "event": "summary",
            "run_dir": str(out_dir),
            "log_path": str(log_path),
            "overall": {
                "n": len(rows_out),
                "correct": sum(1 for r in rows_out if r.get("predicted_color") == r.get("expected_color")),
                "predictions": dict(Counter(r.get("predicted_color") for r in rows_out)),
            },
            "by_expected": {},
        }
        for expected in ("red", "green"):
            subset = [r for r in rows_out if r.get("expected_color") == expected]
            summary["by_expected"][expected] = {
                "n": len(subset),
                "correct": sum(1 for r in subset if r.get("predicted_color") == expected),
                "predictions": dict(Counter(r.get("predicted_color") for r in subset)),
            }
        emit(summary)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
