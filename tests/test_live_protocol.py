"""nav-render/v0 on the wire: the shapes, the client, and the pool.

The protocol is shared with another repository (SimWorld2 carries the service
half), so nothing here may drift silently. Three layers of claim, each of
which someone could break by accident:

1. **the bytes** -- serialising the dataclasses reproduces the golden
   fixtures exactly, byte for byte. The same fixture files are copied into
   SimWorld2, so this one assertion is what pins the two repos together;
2. **the client** -- against a real HTTP server, not a mock, because the
   failure modes worth having (timeouts, error bodies, refused connections)
   only exist on a socket;
3. **the pool** -- least-loaded dispatch, failover mid-call, quarantine on
   the second consecutive health failure, readmission on a good probe, and
   the statefile that carries the verdict across a trainer restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodiedbench.runtime.live.client import (
    RenderServiceError,
    ServiceBusy,
    ServiceUnreachable,
    UERenderClient,
)
from embodiedbench.runtime.live.pool import (
    NoHealthyInstance,
    EndpointsError,
    RenderPool,
)
from embodiedbench.runtime.live.protocol import (
    PROTOCOL,
    CameraSpec,
    Healthz,
    ProtocolViolation,
    RenderBatch,
    RenderItem,
    RenderResponse,
    WireError,
    dumps,
)

from live_stub import FakeRenderService, write_endpoints

GOLDEN = Path(__file__).resolve().parent / "golden" / "nav_render_v0"


def batch_of(*keys: str, episode: str = "ep-test", mode: str = "path") -> RenderBatch:
    return RenderBatch(
        episode_id=episode, return_mode=mode,
        camera=CameraSpec(64, 48, 90.0),
        requests=tuple(
            RenderItem(key=key, x_cm=0.0, y_cm=0.0, z_cm=160.0, yaw_deg=0.0,
                       render_kind="street_view")
            for key in keys
        ),
    )


@pytest.fixture()
def service(tmp_path):
    stub = FakeRenderService(tmp_path / "svc").start()
    yield stub
    stub.stop()


# ─────────────────────────────────────────────────────────────────────────────


class TestTheGoldenBytes:
    """Serialisation must reproduce the shared fixtures exactly.

    "Byte-exact" is the compatibility contract with the SimWorld2 branch: the
    same files sit in both repos, and a formatting change that looks cosmetic
    here is a fixture mismatch there.
    """

    def test_the_render_request_round_trips_byte_exactly(self):
        golden = (GOLDEN / "render_request.json").read_text()
        batch = RenderBatch.from_dict(json.loads(golden))
        assert dumps(batch.to_dict()) == golden

    def test_the_render_response_round_trips_byte_exactly(self):
        golden = (GOLDEN / "render_response.json").read_text()
        response = RenderResponse.from_dict(json.loads(golden))
        assert dumps(response.to_dict()) == golden

    def test_the_healthz_response_round_trips_byte_exactly(self):
        golden = (GOLDEN / "healthz_response.json").read_text()
        health = Healthz.from_dict(json.loads(golden))
        assert dumps(health.to_dict()) == golden

    def test_the_error_response_round_trips_byte_exactly(self):
        golden = (GOLDEN / "error_response.json").read_text()
        error = WireError.from_dict(json.loads(golden))
        assert dumps(error.to_dict()) == golden

    def test_the_request_example_carries_all_three_render_kinds(self):
        """The fixture is only a protocol pin if it exercises the whole
        protocol: a street view, a lamp with its per-request camera, and an
        obstacle with its dressing spec."""
        batch = RenderBatch.from_dict(
            json.loads((GOLDEN / "render_request.json").read_text()))
        kinds = [item.render_kind for item in batch.requests]
        assert kinds == ["street_view", "lamp", "obstacle"]
        lamp = batch.requests[1]
        assert lamp.signal is not None and lamp.signal.state == "red"
        assert lamp.camera == CameraSpec(1280, 960, 40.0)
        obstacle = batch.requests[2]
        assert obstacle.obstacle is not None
        assert obstacle.obstacle.kind == "road_block"
        assert obstacle.obstacle.viewpoint == "carriageway"

    def test_pitch_deg_zero_is_omitted_so_the_golden_bytes_cannot_move(self):
        """The field postdates the fixtures. Serialised-only-when-nonzero is
        what lets both repos gain it without either set of golden bytes
        changing -- an item parsed from the fixture must carry pitch 0.0 and
        serialise without the key."""
        batch = RenderBatch.from_dict(
            json.loads((GOLDEN / "render_request.json").read_text()))
        assert all(item.pitch_deg == 0.0 for item in batch.requests)
        assert all("pitch_deg" not in item.to_dict() for item in batch.requests)

    def test_a_nonzero_pitch_deg_round_trips(self):
        item = RenderItem(key="n0/toward_n1", x_cm=1.0, y_cm=2.0, z_cm=165.0,
                          yaw_deg=48.1, render_kind="lamp", pitch_deg=12.5)
        data = item.to_dict()
        assert data["pitch_deg"] == 12.5
        assert RenderItem.from_dict(data) == item

    def test_a_wrong_protocol_string_is_refused_not_guessed_at(self):
        data = json.loads((GOLDEN / "render_request.json").read_text())
        data["protocol"] = "nav-render/v1"
        with pytest.raises(ProtocolViolation):
            RenderBatch.from_dict(data)

    def test_an_unknown_return_mode_is_refused(self):
        data = json.loads((GOLDEN / "render_request.json").read_text())
        data["return_mode"] = "shared_memory"
        with pytest.raises(ProtocolViolation):
            RenderBatch.from_dict(data)


class TestTheClientOverRealHTTP:
    def test_healthz_speaks_the_protocol(self, service):
        health = UERenderClient(service.base_url).healthz()
        assert health.status == "ok"
        assert health.protocol == PROTOCOL
        assert health.map_name == "citycore-paris"

    def test_a_path_mode_render_returns_openable_frames(self, service):
        from PIL import Image

        results = UERenderClient(service.base_url).render(batch_of("n0/toward_n1"))
        assert len(results) == 1 and results[0].ok
        with Image.open(results[0].path) as image:
            assert image.size == (64, 48)

    def test_a_base64_mode_render_carries_the_png_inline(self, service):
        import base64
        import io

        from PIL import Image

        results = UERenderClient(service.base_url).render(
            batch_of("n0/toward_n1", mode="base64"))
        assert results[0].path is None
        png = base64.b64decode(results[0].png_base64)
        with Image.open(io.BytesIO(png)) as image:
            assert image.size == (64, 48)

    def test_a_failed_item_does_not_kill_the_batch(self, service):
        """The spec's rule: the batch never half-dies. One bad frame comes
        back as a failed result beside the good ones."""
        service.fail_keys = {"n0/toward_bad"}
        results = UERenderClient(service.base_url).render(
            batch_of("n0/toward_ok", "n0/toward_bad"))
        by_key = {r.key: r for r in results}
        assert by_key["n0/toward_ok"].ok
        assert not by_key["n0/toward_bad"].ok
        assert "injected" in by_key["n0/toward_bad"].error

    def test_a_wire_error_becomes_its_own_exception_class(self, service):
        """The taxonomy is what the pool acts on -- ``busy`` fails over,
        ``bad_request`` must not -- so the mapping is a contract, not a
        convenience."""
        service.reject_code = "busy"
        with pytest.raises(ServiceBusy):
            UERenderClient(service.base_url).render(batch_of("n0/toward_n1"))

    def test_a_bare_503_is_busy_even_without_a_protocol_body(self):
        """A saturated service (or a proxy in front of it) may answer 503
        with no spec-shaped body; overload must still be reported as busy,
        not as a generic failure the caller would degrade over."""
        import http.server
        import threading

        class Overloaded(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Overloaded)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            with pytest.raises(ServiceBusy):
                UERenderClient(url).render(batch_of("n0/toward_n1"))
        finally:
            server.shutdown()
            server.server_close()

    def test_a_dead_port_is_unreachable_after_one_reconnect(self, tmp_path):
        # Bind and immediately close a socket so the port is real and refused.
        dead = FakeRenderService(tmp_path / "dead").start()
        url = dead.base_url
        dead.stop()
        with pytest.raises(ServiceUnreachable):
            UERenderClient(url, render_timeout_s=2.0).render(batch_of("k/x"))


class TestThePool:
    def test_it_refuses_to_start_without_an_endpoints_file(self, monkeypatch):
        monkeypatch.delenv("EB_UE_ENDPOINTS", raising=False)
        with pytest.raises(EndpointsError):
            RenderPool()

    def test_the_env_var_names_the_endpoints_file(self, tmp_path, service, monkeypatch):
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        monkeypatch.setenv("EB_UE_ENDPOINTS", str(endpoints))
        results = RenderPool().render(batch_of("n0/toward_n1"))
        assert results[0].ok

    def test_dispatch_spreads_over_instances_rather_than_hammering_one(
            self, tmp_path):
        """Least-loaded with a round-robin tiebreak: sequential batches from
        one caller are always a tie on in-flight, so the tiebreak is what
        actually spreads the work."""
        first = FakeRenderService(tmp_path / "a", instance_id="ue-a").start()
        second = FakeRenderService(tmp_path / "b", instance_id="ue-b").start()
        try:
            pool = RenderPool(write_endpoints(tmp_path / "endpoints.json",
                                              [first, second]))
            for index in range(4):
                pool.render(batch_of(f"n0/toward_{index}"))
            assert len(first.batches) == 2
            assert len(second.batches) == 2
        finally:
            first.stop()
            second.stop()

    def test_a_dead_instance_costs_a_failover_not_an_episode(self, tmp_path):
        live = FakeRenderService(tmp_path / "live", instance_id="ue-live").start()
        dead = FakeRenderService(tmp_path / "dead", instance_id="ue-dead").start()
        dead_url = dead.base_url
        dead.stop()
        try:
            pool = RenderPool(
                write_endpoints(tmp_path / "endpoints.json",
                                [live, ("ue-dead", dead_url)]),
                render_timeout_s=2.0)
            # Whichever instance is picked first, every batch must come back.
            for index in range(3):
                results = pool.render(batch_of(f"n0/toward_{index}"))
                assert results[0].ok
        finally:
            live.stop()

    def test_two_health_failures_quarantine_and_a_good_probe_readmits(
            self, tmp_path, service):
        """The spec's two-strikes rule, end to end: degraded after one failed
        probe, quarantined after the second, readmitted the moment a probe
        succeeds -- and never process-killed, which this test shows by the
        same service object coming straight back."""
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))

        service.healthz_ok = False
        assert pool.check_health() == {service.instance_id: "degraded"}
        assert pool.check_health() == {service.instance_id: "quarantined"}
        with pytest.raises(NoHealthyInstance):
            pool.render(batch_of("n0/toward_n1"))

        service.healthz_ok = True
        assert pool.check_health() == {service.instance_id: "ok"}
        assert pool.render(batch_of("n0/toward_n1"))[0].ok

    def test_render_readmits_a_quarantined_instance_by_probing_it(
            self, tmp_path, service):
        """Quarantine is a pause, not a verdict: when nothing is admitted,
        ``render`` itself probes the quarantined rather than failing while a
        healthy instance sits in the corner."""
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        service.healthz_ok = False
        pool.check_health()
        pool.check_health()
        service.healthz_ok = True
        assert pool.render(batch_of("n0/toward_n1"))[0].ok

    def test_the_verdict_is_written_beside_the_endpoints_file(
            self, tmp_path, service):
        """...and survives a trainer restart: a new pool over the same files
        starts with the quarantine it would otherwise spend two probes
        rediscovering."""
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        pool = RenderPool(endpoints)
        service.healthz_ok = False
        pool.check_health()
        pool.check_health()

        statefile = tmp_path / "endpoints.quarantine.json"
        assert statefile.exists()
        assert json.loads(statefile.read_text())["quarantined"] == [service.instance_id]

        reborn = RenderPool(endpoints)
        with pytest.raises(NoHealthyInstance):
            reborn.render(batch_of("n0/toward_n1"))

    def test_busy_fails_over_to_the_next_instance_without_a_strike(self, tmp_path):
        """Busy is a load signal, not a health verdict (spec section 3): the
        batch lands on the other instance immediately, and the saturated one
        keeps a clean record -- no strike, no quarantine."""
        loaded = FakeRenderService(tmp_path / "a", instance_id="ue-loaded").start()
        spare = FakeRenderService(tmp_path / "b", instance_id="ue-spare").start()
        loaded.busy_batches = 10**6
        try:
            pool = RenderPool(write_endpoints(tmp_path / "endpoints.json",
                                              [loaded, spare]))
            pool.busy_backoff_s = (0.01, 0.01, 0.01)
            results = pool.render(batch_of("n0/toward_n1"))
            assert results[0].ok
            assert len(spare.batches) == 1
            by_id = {m.id: m for m in pool.members}
            assert by_id["ue-loaded"].strikes == 0
            assert not by_id["ue-loaded"].quarantined
        finally:
            loaded.stop()
            spare.stop()

    def test_an_all_busy_sweep_backs_off_and_retries_before_giving_up(
            self, tmp_path, service):
        """One transient busy costs a bounded wait, not the batch: the pool
        sleeps and sweeps again, and the retry lands on the same -- still
        healthy -- instance."""
        service.busy_batches = 1
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        pool.busy_backoff_s = (0.01, 0.02, 0.04)
        results = pool.render(batch_of("n0/toward_n1"))
        assert results[0].ok
        assert service.busy_hits == 1
        assert pool.members[0].strikes == 0
        assert not pool.members[0].quarantined

    def test_a_persistently_busy_fleet_raises_service_busy_not_death(
            self, tmp_path, service):
        """After the bounded retries the batch is given up as ServiceBusy --
        the caller's signal to skip and retry on the next lookup -- and the
        instance is still admitted: busy never becomes a quarantine."""
        service.busy_batches = 10**6
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        pool.busy_backoff_s = (0.01, 0.02, 0.04)
        with pytest.raises(ServiceBusy):
            pool.render(batch_of("n0/toward_n1"))
        # The initial sweep plus one per backoff step, each answered busy.
        assert service.busy_hits == 1 + len(pool.busy_backoff_s)
        assert pool.members[0].strikes == 0
        assert not pool.members[0].quarantined
        # The fleet drains and the same pool serves the very next batch.
        service.busy_batches = 0
        assert pool.render(batch_of("n0/toward_n2"))[0].ok

    def test_a_render_transport_death_counts_as_strikes_too(self, tmp_path):
        """A failed batch is the same evidence a probe would have gathered,
        arriving earlier; after two of them the pool stops offering work
        without waiting for anyone to call check_health."""
        doomed = FakeRenderService(tmp_path / "doomed", instance_id="ue-doomed").start()
        url = doomed.base_url
        doomed.stop()
        pool = RenderPool(
            write_endpoints(tmp_path / "endpoints.json", [("ue-doomed", url)]),
            render_timeout_s=2.0)
        for _ in range(2):
            with pytest.raises(RenderServiceError):
                pool.render(batch_of("k/x"))
        assert pool.members[0].quarantined
