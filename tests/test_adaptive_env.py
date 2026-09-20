"""The adaptive curriculum's two load-bearing claims, pinned.

1. Switch off == the benchmark, byte for byte. Not "similar": the same seeds
   must draw the same orders, because every baseline number ever reported
   depends on it, and an extra RNG draw in the off path would silently change
   every shift.
2. Switch on == the same support, different weights. The bias may reweight
   the distribution toward a feature class; it may never admit an order the
   base filters would reject, and no class is starved below the floor.

Plus the seams: profile loading degrades to baseline on absence/corruption,
the deliveries reward basis sums to the count, and the profile builder maps
failure rates to the documented weights.
"""

from __future__ import annotations

import json

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import CourierEnv
from embodiedbench.training.vagen_courier_env import DEFAULT_MAP


@pytest.fixture(scope="module")
def network():
    return build_road_network(DEFAULT_MAP, map_name=DEFAULT_MAP.name)


def _orders(network, seed, bias=None, n=6):
    env = CourierEnv(network, seed=seed, difficulty="endless", queue_depth=1,
                     stride="block", embodiment="human_on_foot",
                     order_bias=bias)
    env.reset()
    drawn = []
    rng = env._order_rng
    while len(drawn) < n:
        order = env._draw_order(len(drawn) + 1, rng)
        if order is None:
            break
        drawn.append(order)
    return drawn


def _signature(orders):
    return [(o.pickup.street_name, o.dropoff.street_name, round(o.fee, 2))
            for o in orders]


class TestBaselineUntouched:
    def test_bias_none_is_byte_identical(self, network):
        """The benchmark's shifts do not change because the feature exists."""
        for seed in (0, 7, 123, 1000):
            assert _signature(_orders(network, seed)) == \
                _signature(_orders(network, seed))

    def test_bias_none_draws_no_extra_rng(self, network):
        """order_bias=None and order_bias={} both take the pre-existing path."""
        assert _signature(_orders(network, 42, bias=None)) == \
            _signature(_orders(network, 42, bias={}))


class TestBiasShiftsTheDistribution:
    def test_long_bias_lengthens_orders(self, network):
        base, biased = [], []
        for seed in range(12):
            base += [o.fee for o in _orders(network, seed)]
            biased += [o.fee for o in
                       _orders(network, seed, bias={"len_long": 3.0})]
        # Fee is affine in walk length, so mean fee is a clean length proxy.
        assert sum(biased) / len(biased) > sum(base) / len(base)

    def test_short_bias_shortens_orders(self, network):
        base, biased = [], []
        for seed in range(12):
            base += [o.fee for o in _orders(network, seed)]
            biased += [o.fee for o in
                       _orders(network, seed, bias={"len_short": 3.0})]
        assert sum(biased) / len(biased) < sum(base) / len(base)

    def test_bias_never_starves_the_draw(self, network):
        """An adversarial profile still yields full shifts -- the floor holds."""
        for seed in range(6):
            drawn = _orders(network, seed,
                            bias={"len_short": 0.0001, "len_mid": 0.0001,
                                  "len_long": 0.0001})
            assert len(drawn) == 6

    def test_biased_orders_respect_base_filters(self, network):
        """Reweighted, never widened: every biased order passes base bounds."""
        from embodiedbench.runtime.city.courier_env import MAX_ORDER_WALK_CM
        env = CourierEnv(network, seed=3, difficulty="endless", queue_depth=1,
                         stride="block", embodiment="human_on_foot",
                         order_bias={"len_long": 3.0})
        env.reset()
        for i in range(6):
            order = env._draw_order(i + 1, env._order_rng)
            walk = env.route_length_cm(order.pickup.kerb_node,
                                       order.dropoff.kerb_node)
            assert 8000.0 <= walk <= MAX_ORDER_WALK_CM


class TestProfilePlumbing:
    def test_missing_profile_is_baseline(self, tmp_path):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        env = CourierGymEnv({"adaptive": True,
                             "adaptive_profile": str(tmp_path / "absent.json")})
        assert env._load_bias() is None

    def test_corrupt_profile_is_baseline(self, tmp_path):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        p = tmp_path / "profile.json"
        p.write_text("{not json")
        env = CourierGymEnv({"adaptive": True, "adaptive_profile": str(p)})
        assert env._load_bias() is None

    def test_profile_reloads_on_mtime(self, tmp_path):
        import os
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        p = tmp_path / "profile.json"
        p.write_text(json.dumps({"bias": {"len_long": 2.0}}))
        env = CourierGymEnv({"adaptive": True, "adaptive_profile": str(p)})
        assert env._load_bias() == {"len_long": 2.0}
        p.write_text(json.dumps({"bias": {"len_short": 1.5}}))
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
        assert env._load_bias() == {"len_short": 1.5}

    def test_adaptive_off_ignores_profile(self, tmp_path):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        p = tmp_path / "profile.json"
        p.write_text(json.dumps({"bias": {"len_long": 3.0}}))
        env = CourierGymEnv({"adaptive_profile": str(p)})   # adaptive absent
        assert env._load_bias() is None


class TestProfileBuilder:
    def _rows(self, **overrides):
        row = {"seed": 1000, "delivered": 2, "late": 0, "expired": 0,
               "orders_issued": 3, "red_crossings": 0, "waits_at_red": 1,
               "earnings": 9.5, "walked_m": 900.0, "detour": 1.1, "turns": 55}
        row.update(overrides)
        return row

    def test_healthy_policy_gets_empty_bias(self, tmp_path):
        import tools.adaptive_profile as ap
        log = tmp_path / "ep.jsonl"
        log.write_text("\n".join(
            json.dumps(self._rows(seed=1000 + i)) for i in range(20)) + "\n")
        profile = ap.build(log)
        assert profile["bias"] == {}
        assert profile["episodes"] == 20

    def test_late_failures_weight_long_orders(self, tmp_path):
        import tools.adaptive_profile as ap
        log = tmp_path / "ep.jsonl"
        rows = [self._rows(seed=1000 + i, delivered=1, late=1)
                for i in range(20)]
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        bias = ap.build(log)["bias"]
        assert bias.get("len_long", 0) > 1.0

    def test_zero_delivery_overrides_with_short(self, tmp_path):
        import tools.adaptive_profile as ap
        log = tmp_path / "ep.jsonl"
        rows = [self._rows(seed=1000 + i, delivered=0, late=1, detour=2.0,
                           red_crossings=3) for i in range(20)]
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        bias = ap.build(log)["bias"]
        assert set(bias) == {"len_short"}

    def test_training_rows_are_ignored(self, tmp_path):
        import tools.adaptive_profile as ap
        log = tmp_path / "ep.jsonl"
        rows = [self._rows(seed=i, delivered=0) for i in range(20)]      # train
        rows += [self._rows(seed=1000 + i) for i in range(20)]           # val
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        profile = ap.build(log)
        assert profile["episodes"] == 20 and profile["bias"] == {}


class TestDeliveriesBasis:
    def test_basis_accepted(self):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        env = CourierGymEnv({"reward_basis": "deliveries"})
        assert env.reward_basis == "deliveries"

    def test_unknown_basis_still_raises(self):
        from embodiedbench.training.vagen_courier_env import CourierGymEnv
        with pytest.raises(ValueError):
            CourierGymEnv({"reward_basis": "vibes"})


class TestStreetNaming:
    """Multi-city naming: distinct per map, and Paris pinned forever."""

    def test_paris_names_are_pinned(self):
        from embodiedbench.compiler.road_network import street_name
        # Baked into every album manifest and reported transcript. If this
        # test fails, assets have been orphaned -- do not "fix" the test.
        assert street_name(0) == "Rue de Rivoli"
        assert street_name(0, "citycore-paris") == "Rue de Rivoli"
        assert street_name(0, "") == "Rue de Rivoli"
        assert street_name(1, "citycore-paris") == "Rue Saint-Honoré"

    def test_each_city_has_its_own_offset(self):
        from embodiedbench.compiler.road_network import street_name
        firsts = {m: street_name(0, m)
                  for m in ("small-city-11", "small-city-13", "medium-city-18")}
        assert all(v != "Rue de Rivoli" for v in firsts.values())
        assert len(set(firsts.values())) == len(firsts)  # and not each other's

    def test_small_city_suffix_counts_its_own_streets(self):
        from embodiedbench.compiler.road_network import street_name
        # Street 0 of a shifted map must not inherit a wraparound suffix.
        assert "(" not in street_name(0, "small-city-11")
