"""The M1 walking skeleton runs through every layer, deterministically.

the design plan M1 accepts when "the walking-skeleton command exits zero from a clean
environment and three runs produce identical trajectory and score hashes".

"Clean environment" is taken literally: the command is invoked as a subprocess
with a fresh interpreter and no inherited ``PYTHONHASHSEED``, so a result that
depends on warm module state or on string-hash ordering cannot pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from embodiedbench.schemas.environment import EnvironmentBundle
from embodiedbench.schemas.episode import EpisodeSpec
from embodiedbench.schemas.trajectory import ScoreReport, Trajectory
from embodiedbench.schemas.world import WorldBundle

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "vendor" / "vagen").exists(),
    reason="vendored VAGEN checkout not present",
)


def _run_skeleton(tmp_path: Path, *extra: str) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "embodiedbench.skeleton", "--out", str(tmp_path), *extra],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_skeleton_exits_zero_from_a_clean_process(tmp_path):
    code, output = _run_skeleton(tmp_path)
    assert code == 0, output[-4000:]


def test_three_clean_runs_produce_identical_hashes(tmp_path):
    code, output = _run_skeleton(tmp_path, "--repeat", "3")
    assert code == 0, output[-4000:]
    assert "all 3 runs produced identical hashes" in output


def test_skeleton_writes_every_layers_artifact(tmp_path):
    code, output = _run_skeleton(tmp_path)
    assert code == 0, output[-4000:]
    for name in (
        "world_bundle",
        "environment_bundle",
        "episode_spec",
        "trajectory",
        "score_report",
        "summary",
    ):
        assert (tmp_path / f"{name}.json").exists(), f"missing {name}.json"


def test_every_written_artifact_revalidates_against_its_schema(tmp_path):
    """An artifact that cannot be read back is not an artifact."""
    code, output = _run_skeleton(tmp_path)
    assert code == 0, output[-4000:]
    for name, model in (
        ("world_bundle", WorldBundle),
        ("environment_bundle", EnvironmentBundle),
        ("episode_spec", EpisodeSpec),
        ("trajectory", Trajectory),
        ("score_report", ScoreReport),
    ):
        payload = json.loads((tmp_path / f"{name}.json").read_text())
        restored = model.from_dict(payload)
        assert restored.to_dict() == payload


def test_summary_reports_a_completed_delivery(tmp_path):
    """A skeleton that never reaches the task logic proves nothing about it."""
    code, output = _run_skeleton(tmp_path)
    assert code == 0, output[-4000:]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["steps"] > 5
    assert summary["deliveries"] >= 1
    assert summary["success"] is True
    assert summary["nodes"] > 0 and summary["edges"] > 0


def test_trajectory_links_to_the_environment_it_ran_in(tmp_path):
    code, _ = _run_skeleton(tmp_path)
    assert code == 0
    trajectory = Trajectory.from_dict(json.loads((tmp_path / "trajectory.json").read_text()))
    spec = EpisodeSpec.from_dict(json.loads((tmp_path / "episode_spec.json").read_text()))
    world = WorldBundle.from_dict(json.loads((tmp_path / "world_bundle.json").read_text()))
    assert trajectory.instance_id == spec.instance_id
    assert trajectory.environment_id == spec.environment_id
    # design plan §12.1: the environment hash is the benchmark's "base commit".
    assert spec.environment_sha256 == world.content_hash()


def test_score_report_declines_to_invent_the_primary_metric(tmp_path):
    """design plan §12.5 defines it against an upper-bound policy that does not exist yet."""
    code, _ = _run_skeleton(tmp_path)
    assert code == 0
    score = ScoreReport.from_dict(json.loads((tmp_path / "score_report.json").read_text()))
    assert score.normalized_utility_vs_upper_bound is None
    assert score.upper_bound_undefined_reason


def test_generated_episode_spec_is_reproducible_across_processes():
    """the design plan M4 in miniature: same (world, config, seed) -> identical spec."""
    code_a, out_a = _run_once_spec_hash()
    code_b, out_b = _run_once_spec_hash()
    assert code_a == code_b == 0
    assert out_a == out_b


def _run_once_spec_hash() -> tuple[int, str]:
    script = (
        "import json,contextlib,io\n"
        "from embodiedbench.skeleton import compile_world\n"
        "from embodiedbench.tasks.delivery import DeliveryTask\n"
        "buf=io.StringIO()\n"
        "t=DeliveryTask('standard')\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    w,e,spec=compile_world('small-city-11', seed=7)\n"
        "    ep=t.generate(spec, seed=7, world=w)\n"
        "print(json.dumps({'world': w.content_hash(), 'env_spec': spec.content_hash(),"
        " 'episode': ep.content_hash()}))\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, check=False,
    )
    return proc.returncode, proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-2000:]
