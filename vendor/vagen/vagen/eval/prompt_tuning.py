"""
Standalone prompt-tuning eval harness.

The model is served by a persistent SGLang server (launched once, kept running).
This script connects to it via the OpenAI-compatible HTTP API and directly drives
environment instances — no Ray, no FSDP, no verl training stack involved.

Typical workflow
----------------
# 1. Launch SGLang server ONCE (keep it running in a separate terminal):
#    python -m sglang.launch_server \\
#        --model-path ~/models/Qwen2.5-VL-3B-Instruct \\
#        --port 30000 --chat-template qwen2-vl

# 2. Edit your prompt in:
#    vagen/envs/deliverybench/utils/prompt.py   (prompt text)
#    examples/deliverybench/val_deliverybench.yaml  (prompt_format: free_think|wm)

# 3. Run eval (fast — no model reload):
#    python -m vagen.eval.prompt_tuning \\
#        --model-path ~/models/Qwen2.5-VL-3B-Instruct \\
#        --val-yaml examples/deliverybench/val_deliverybench.yaml
"""

import argparse
import asyncio
import base64
import importlib
import io
import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import AsyncOpenAI
from PIL import Image

from vagen.gym_agent_dataset import load_envspecs, _generate_seeds_for_spec

# Default path to the env registry relative to the repo root
_DEFAULT_REGISTRY = Path(__file__).parent.parent / "configs" / "env_registry.yaml"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_env_registry(registry_path: str) -> Dict[str, Any]:
    """Load environment class registry from YAML and import the classes."""
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    registry: Dict[str, Any] = {}
    for name, cls_path in data["env_registry"].items():
        module_path, class_name = cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        registry[name] = getattr(module, class_name)
    return registry


def _pil_to_data_uri(img: Image.Image) -> str:
    img = img.resize((img.width // 4, img.height // 4))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def obs_to_content(obs: Dict[str, Any]):
    """
    Convert an environment observation dict to an OpenAI message content value.

    Observations follow the GymImageEnv protocol:
        {
            "obs_str": "...<image>...",
            "multi_modal_input": {"<image>": [PIL.Image, ...]}
        }

    When images are present the content is a list of typed parts so that
    vision models (e.g. Qwen2.5-VL) receive them correctly.  When there are
    no images a plain string is returned for efficiency.
    """
    text: str = obs.get("obs_str", "") or ""
    images: List[Image.Image] = (
        obs.get("multi_modal_input", {}).get("<image>", []) or []
    )

    if not images:
        return text

    # Interleave image parts and text segments split by the <image> placeholder
    parts: List[Dict[str, Any]] = []
    segments = text.split("<image>")
    for i, segment in enumerate(segments):
        if segment:
            parts.append({"type": "text", "text": segment})
        if i < len(images):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_uri(images[i])},
                }
            )
    if not parts:
        parts = [{"type": "text", "text": ""}]
    return parts


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _extract_images(obs: Dict[str, Any]) -> List[Image.Image]:
    """Extract PIL images from an observation dict."""
    return list(obs.get("multi_modal_input", {}).get("<image>", []) or [])


def _fmt_hours(h: float) -> str:
    """Format fractional hours as '1h 23m'."""
    total_min = round(h * 60)
    return f"{total_min // 60}h {total_min % 60:02d}m"


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    n_turns: int
    data_source: str
    seed: int
    env_name: str
    sim_hours: float = 0.0
    error: Optional[str] = None
    trajectory: Optional[List[Dict[str, Any]]] = None
    images_per_turn: Optional[List[List[Image.Image]]] = None


def _strip_images(content: Any) -> Any:
    """Strip base64 image data from message content for lightweight trajectory logs."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": "<image_omitted>"}})
            else:
                parts.append(p)
        return parts
    return content


async def run_episode(
    client: AsyncOpenAI,
    model_path: str,
    env_cls: Any,
    env_config: Dict[str, Any],
    seed: int,
    max_turns: int,
    data_source: str,
    sampling_params: Dict[str, Any],
    save_images: bool = False,
) -> EpisodeResult:
    """Run a single episode and return metrics."""
    env = env_cls(env_config=env_config)
    total_reward = 0.0
    n_turns = 0
    success = False
    sim_hours = 0.0
    error_msg: Optional[str] = None
    trajectory: List[Dict[str, Any]] = []
    images_per_turn: List[List[Image.Image]] = []

    try:
        init_obs, _ = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        messages: List[Dict[str, Any]] = []
        if sys_obs:
            messages.append({"role": "system", "content": obs_to_content(sys_obs)})
        if init_obs:
            messages.append({"role": "user", "content": obs_to_content(init_obs)})

        last_obs_str = (init_obs or {}).get("obs_str", "")
        last_obs_images = _extract_images(init_obs or {}) if save_images else []

        for _ in range(max_turns):
            response = await client.chat.completions.create(
                model=model_path,
                messages=messages,
                **sampling_params,
            )
            action: str = response.choices[0].message.content or ""
            usage = response.usage

            turn_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            turn_cached_tokens = (
                details.get("cached_tokens", 0) if isinstance(details, dict)
                else getattr(details, "cached_tokens", 0) or 0
            ) if details else 0
            turn_cache_hit_rate = (
                turn_cached_tokens / turn_prompt_tokens
                if turn_prompt_tokens > 0 else 0.0
            )

            messages.append({"role": "assistant", "content": action})
            n_turns += 1

            obs, reward, done, info = await env.step(action)
            total_reward += float(reward)
            success = bool(info.get("traj_success", info.get("success", False)))

            traj_metrics = (info.get("metrics") or {}).get("traj_metrics", {})
            sim_hours = float(traj_metrics.get("sim_hours", 0.0))

            trajectory.append({
                "turn": n_turns,
                "observation": last_obs_str,
                "action": action,
                "reward": float(reward),
                "cumulative_reward": float(total_reward),
                "done": bool(done),
                "success": success,
                "sim_time": _fmt_hours(sim_hours),
                "prompt_tokens": turn_prompt_tokens,
                "cached_tokens": turn_cached_tokens,
                "cache_hit_rate": turn_cache_hit_rate,
                "info": {
                    k: v for k, v in (info or {}).items()
                    if k not in ("raw_info",)
                },
            })
            images_per_turn.append(last_obs_images)

            if done or success:
                break

            user_content = obs_to_content(obs)
            messages.append({"role": "user", "content": user_content})
            last_obs_str = (obs or {}).get("obs_str", "")
            last_obs_images = _extract_images(obs or {}) if save_images else []

    except Exception:
        error_msg = traceback.format_exc()
    finally:
        try:
            await env.close()
        except Exception:
            pass

    traj_with_messages = {
        "seed": seed,
        "env_name": env_cls.__name__,
        "data_source": data_source,
        "n_turns": n_turns,
        "reward": total_reward,
        "success": success,
        "sim_time": _fmt_hours(sim_hours),
        "error": error_msg,
        "turns": trajectory,
    }

    return EpisodeResult(
        reward=total_reward,
        success=success,
        n_turns=n_turns,
        data_source=data_source,
        seed=seed,
        env_name=env_cls.__name__,
        sim_hours=sim_hours,
        error=error_msg,
        trajectory=traj_with_messages,
        images_per_turn=images_per_turn,
    )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

async def eval_prompts(
    server_url: str,
    model_path: str,
    val_yaml: str,
    n_per_env: int = 1,
    max_concurrent: int = 20,
    temperature: float = 0.0,
    max_tokens: int = 512,
    output_dir: Optional[str] = None,
    registry_path: Optional[str] = None,
    base_seed: int = 0,
    save_images: bool = False,
) -> List[EpisodeResult]:
    registry_path = registry_path or str(_DEFAULT_REGISTRY)
    env_registry = load_env_registry(registry_path)

    env_specs = load_envspecs(val_yaml).specs

    # Expand specs into individual (env_cls, env_config, seed, max_turns, data_source) tuples
    episodes: List[Tuple] = []
    seeds_by_spec: Dict[str, List[int]] = {}
    for spec_idx, spec in enumerate(env_specs):
        env_cls = env_registry.get(spec.name)
        if env_cls is None:
            raise ValueError(
                f"Environment '{spec.name}' not found in registry {registry_path}. "
                f"Available: {list(env_registry.keys())}"
            )
        seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
        seeds_by_spec[spec.name] = seeds
        print(f"[{spec.name}] seeds ({len(seeds)}): {seeds}")
        env_config = dict(spec.config) if spec.config else {}
        max_turns = spec.max_turns
        data_source = spec.data_source

        for seed in seeds:
            for _ in range(n_per_env):
                episodes.append((env_cls, env_config, seed, max_turns, data_source))

    client = AsyncOpenAI(api_key="none", base_url=f"{server_url.rstrip('/')}/v1")
    sampling_params: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    sem = asyncio.Semaphore(max_concurrent)

    async def bounded_run(ep: Tuple) -> EpisodeResult:
        env_cls, env_config, seed, max_turns, data_source = ep
        async with sem:
            return await run_episode(
                client=client,
                model_path=model_path,
                env_cls=env_cls,
                env_config=env_config,
                seed=seed,
                max_turns=max_turns,
                data_source=data_source,
                sampling_params=sampling_params,
                save_images=save_images,
            )

    print(
        f"Running {len(episodes)} episodes "
        f"({len(env_specs)} env spec(s), n_per_env={n_per_env}, "
        f"max_concurrent={max_concurrent}) ..."
    )
    results: List[EpisodeResult] = await asyncio.gather(
        *[bounded_run(ep) for ep in episodes]
    )

    # --- aggregate ---
    errors = [r for r in results if r.error]
    if errors:
        print(f"\nWARNING: {len(errors)} episode(s) raised exceptions:")
        for r in errors[:5]:
            print(f"  env={r.env_name} seed={r.seed}\n  {r.error.splitlines()[-1]}")

    by_source: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by_source.setdefault(r.data_source, []).append(r)

    def _cache_stats(rlist: List[EpisodeResult]) -> Tuple[int, int, float]:
        prompt = sum(
            t.get("prompt_tokens", 0)
            for r in rlist for t in (r.trajectory or {}).get("turns", [])
        )
        cached = sum(
            t.get("cached_tokens", 0)
            for r in rlist for t in (r.trajectory or {}).get("turns", [])
        )
        rate = cached / prompt if prompt > 0 else 0.0
        return prompt, cached, rate

    print()
    for source, rlist in by_source.items():
        n = len(rlist)
        mean_reward = sum(r.reward for r in rlist) / n
        success_rate = sum(r.success for r in rlist) / n
        mean_turns = sum(r.n_turns for r in rlist) / n
        mean_sim_hours = sum(r.sim_hours for r in rlist) / n
        src_prompt, src_cached, src_cache_rate = _cache_stats(rlist)

        print(f"[{source}]  n={n}")
        print(f"  reward          : {mean_reward:.4f}")
        print(f"  success_rate    : {success_rate:.4f}")
        print(f"  mean_turns      : {mean_turns:.2f}")
        print(f"  mean_sim_time   : {_fmt_hours(mean_sim_hours)}")
        print(f"  prompt_tokens   : {src_prompt}")
        print(f"  cached_tokens   : {src_cached}")
        print(f"  cache_hit_rate  : {src_cache_rate:.4f}")

    total_prompt, total_cached, overall_cache_rate = _cache_stats(results)
    print(f"\n[Overall Cache Stats]")
    print(f"  total_prompt_tokens : {total_prompt}")
    print(f"  total_cached_tokens : {total_cached}")
    print(f"  cache_hit_rate      : {overall_cache_rate:.4f}")

    if output_dir:
        from datetime import datetime as _dt
        run_name = f"run_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(output_dir, run_name)
        traj_dir = os.path.join(run_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        results_path = os.path.join(run_dir, "results.json")
        serializable = {
            "seeds": seeds_by_spec,
            "cache_stats": {
                "total_prompt_tokens": total_prompt,
                "total_cached_tokens": total_cached,
                "cache_hit_rate": overall_cache_rate,
            },
            "episodes": [
                {
                    "env_name": r.env_name,
                    "data_source": r.data_source,
                    "seed": r.seed,
                    "reward": r.reward,
                    "success": r.success,
                    "n_turns": r.n_turns,
                    "sim_time": _fmt_hours(r.sim_hours),
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {results_path}")

        for i, r in enumerate(results):
            ep_name = f"ep{i:04d}_seed{r.seed}"
            if r.trajectory is not None:
                traj_path = os.path.join(traj_dir, f"{ep_name}.json")
                with open(traj_path, "w") as f:
                    json.dump(r.trajectory, f, indent=2, default=str)

            if r.images_per_turn:
                img_dir = os.path.join(traj_dir, f"{ep_name}_images")
                os.makedirs(img_dir, exist_ok=True)
                for turn_idx, imgs in enumerate(r.images_per_turn):
                    for img_idx, img in enumerate(imgs):
                        img_path = os.path.join(
                            img_dir,
                            f"turn{turn_idx + 1:03d}_img{img_idx}.png",
                        )
                        img.save(img_path)
        print(f"Trajectories saved to {traj_dir}/")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt tuning eval: SGLang server + direct env interaction."
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:30000",
        help="Base URL of the running SGLang server (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model path / name passed to the SGLang server (must match what the server loaded)",
    )
    parser.add_argument(
        "--val-yaml",
        required=True,
        help="Path to the validation env spec YAML (e.g. examples/deliverybench/val_deliverybench.yaml)",
    )
    parser.add_argument(
        "--n-per-env",
        type=int,
        default=1,
        help="Number of episodes per seed (default: 1)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum number of concurrent episodes (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 = greedy (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per turn (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base directory for saving results; a timestamped run folder is created inside",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to env_registry.yaml (default: vagen/configs/env_registry.yaml)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed for deterministic seed generation (default: 0)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Save map images for each turn in the trajectory (default: off)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        eval_prompts(
            server_url=args.server_url,
            model_path=args.model_path,
            val_yaml=args.val_yaml,
            n_per_env=args.n_per_env,
            max_concurrent=args.max_concurrent,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            registry_path=args.registry,
            base_seed=args.base_seed,
            save_images=args.save_images,
        )
    )


if __name__ == "__main__":
    main()
