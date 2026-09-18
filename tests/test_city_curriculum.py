"""The city curriculum: three modes of steering WHERE the courier trains.

Same restraint story as the order-class curriculum: deterministic per seed,
30%-capped by the mixing coin, degrades to uniform rotation on any missing
or malformed profile -- and the synthetic-log tests pin each mode's logic
(learnability finds the frontier city; the ladder unlocks by delivery rate;
the battery skill-lean appears below the bar and decays above mastery).
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import adaptive_profile as ap  # noqa: E402

MAPS = REPO / "vendor" / "vagen" / "vagen" / "envs" / "deliverybench" / "maps"
needs_maps = pytest.mark.skipif(not (MAPS / "citycore-paris").exists(),
                                reason="vendored maps not present")


def synth_log(tmp_path: Path, alive_rate: float = 0.66) -> Path:
    """Three cities: one solved, one frontier, one hopeless; val at 60%."""
    rng = random.Random(3)
    rows = []
    for seed in range(120):
        city = ["small-city-11", "citycore-paris", "large-city-26"][seed % 3]
        for _ in range(8):
            if city == "small-city-11":
                money = 14 + rng.random() * 0.3          # solved: no variance
            elif city == "citycore-paris":
                money = rng.choice([0.0, 12.0]) + rng.random()  # frontier
            else:
                money = 0.0                               # hopeless
            rows.append({"seed": seed, "city": city, "earnings": round(money, 2),
                         "delivered": int(money > 1), "late": 0,
                         "red_crossings": 0, "detour": 1.2, "order_classes": {}})
    for i, seed in enumerate(range(1000, 1064)):
        rows.append({"seed": seed, "delivered": int(i % 10 < 6), "late": 0,
                     "red_crossings": 0, "earnings": 8.0, "detour": 1.1,
                     "phone_alive": (i / 64.0) < alive_rate, "recharges": 0})
    log = tmp_path / "episodes.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows))
    return log


class TestBuilderModes:
    def test_city_mode_weights_the_frontier_only(self, tmp_path):
        prof = ap.build(synth_log(tmp_path), mode="city")
        assert set(prof["city_weights"]) == {"citycore-paris"}
        assert prof["city_weights"]["citycore-paris"] > 1.2
        assert prof["bias"] == {}

    def test_city_mode_ema_converges(self, tmp_path):
        log = synth_log(tmp_path)
        prof = ap.build(log, mode="city")
        again = ap.build(log, mode="city", previous=prof)
        assert again["city_weights"]["citycore-paris"] > prof["city_weights"]["citycore-paris"]

    def test_city_ladder_unlocks_by_delivery_rate(self, tmp_path):
        prof = ap.build(synth_log(tmp_path), mode="city_ladder")
        cities = prof["city_weights"]
        # 60% delivery unlocks the first five rungs; newest carries the lean.
        assert "small-city-11" in cities and "medium-city-22" in cities
        assert "large-city-26" not in cities
        assert cities["medium-city-22"] == max(cities.values())

    def test_skill_battery_leans_below_the_bar(self, tmp_path):
        prof = ap.build(synth_log(tmp_path, alive_rate=0.66), mode="skill_battery")
        assert prof["bias"].get("len_long", 0) > 1.0
        assert prof["bias"].get("signalled", 0) > 1.0

    def test_skill_battery_decays_at_mastery(self, tmp_path):
        log = synth_log(tmp_path, alive_rate=1.0)
        lean = {"bias": {"len_long": 2.0, "signalled": 2.0}}
        prof = ap.build(log, mode="skill_battery", previous=lean)
        for v in prof["bias"].values():
            assert v < 2.0, "the lean must decay once the skill is mastered"


@pytest.fixture(autouse=True)
def _albums_for_the_dispatcher(monkeypatch):
    """The dispatcher tests reset real cities, so they need the albums.

    The runtime reads ALBUMS_DIR and nothing else (a fallback to the baking
    workstation was removed for the release); where the albums are not
    mounted these tests skip rather than fail.
    """
    import os
    base = os.environ.get("ALBUMS_DIR") or "/data/albums"
    if not (Path(base) / "paris_streets_pavement" / "citycore-paris").is_dir():
        pytest.skip("albums not mounted (set ALBUMS_DIR)")
    monkeypatch.setenv("ALBUMS_DIR", base)


@needs_maps
class TestDispatcherCitySampling:
    ROTATION = ["citycore-paris", "small-city-11", "small-city-13"]

    def env_of(self, profile_path=None, adaptive=True, mix=0.3):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        cfg = {"difficulty": "endless", "hazards": False,
               "map_rotation": list(self.ROTATION), "adaptive_mix": mix}
        if adaptive and profile_path:
            cfg["adaptive"] = True
            cfg["adaptive_profile"] = str(profile_path)
        return CourierGymEnv(env_config=cfg)

    def picks(self, env, n=90):
        out = collections.Counter()
        for seed in range(n):
            asyncio.run(env.reset(seed=seed))
            out[env.map_dir.name] += 1
        return out

    def test_weights_tilt_the_rotation(self, tmp_path):
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({"bias": {}, "city_weights":
                                    {"citycore-paris": 3.0, "small-city-11": 0.5}}))
        picks = self.picks(self.env_of(prof))
        assert picks["citycore-paris"] > picks["small-city-13"]

    def test_same_seed_same_city(self, tmp_path):
        prof = tmp_path / "p.json"
        prof.write_text(json.dumps({"bias": {}, "city_weights":
                                    {"citycore-paris": 3.0}}))
        a, b = self.env_of(prof), self.env_of(prof)
        for seed in (5, 17, 44):
            asyncio.run(a.reset(seed=seed))
            asyncio.run(b.reset(seed=seed))
            assert a.map_dir.name == b.map_dir.name

    def test_no_profile_means_uniform(self):
        picks = self.picks(self.env_of(adaptive=False), n=30)
        assert all(v == 10 for v in picks.values())

    def test_malformed_profile_degrades_to_uniform(self, tmp_path):
        prof = tmp_path / "p.json"
        prof.write_text("{not json")
        picks = self.picks(self.env_of(prof), n=30)
        assert all(v == 10 for v in picks.values())


class TestCityNeed:
    def test_need_mode_weights_the_poorest(self, tmp_path):
        log = synth_log(tmp_path)
        prof = ap.build(log, mode="city_need")
        w = prof["city_weights"]
        # hopeless lc26 (earnings 0) must get the biggest lean; solved
        # sc11 (highest mean) the smallest -- the exact confusion between
        # "hard" and "hopeless" the learnability mode exists to avoid.
        assert w["large-city-26"] == max(w.values())
        # The richest city's weight sits at ~1.0 and is pruned from the dict
        # (absent means neutral) -- it must not carry a lean either way.
        assert w.get("small-city-11", 1.0) <= min(w.values())
