from __future__ import annotations

import io

import pytest
from farm_edge_agent.doctor import network
from farm_edge_agent.doctor.network import Finding, Status


def test_step_dns_ok_when_resolver_returns_addresses() -> None:
    f = network.step_dns("farm.dev", resolver=lambda h: ["192.0.2.1", "192.0.2.2"])
    assert f.status is Status.OK
    assert f.error_code is None
    assert f.detail["addrs"] == ["192.0.2.1", "192.0.2.2"]


def test_step_dns_failed_when_resolver_raises() -> None:
    def boom(_: str) -> list[str]:
        raise OSError("nodename nor servname")

    f = network.step_dns("farm.dev", resolver=boom)
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"
    assert f.fix is not None


def test_step_dns_failed_when_resolver_returns_empty() -> None:
    f = network.step_dns("farm.dev", resolver=lambda h: [])
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


def test_step_ws_upgrade_ok_on_101() -> None:
    f = network.step_ws_upgrade("wss://farm.dev/d", opener=lambda url: 101)
    assert f.status is Status.OK
    assert f.error_code is None


def test_step_ws_upgrade_failed_on_403() -> None:
    f = network.step_ws_upgrade("wss://farm.dev/d", opener=lambda url: 403)
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"
    assert "FARM_RELAY" in (f.fix or "")


def test_step_ws_upgrade_failed_on_exception() -> None:
    def boom(_: str) -> int:
        raise ConnectionError("blocked")

    f = network.step_ws_upgrade("wss://farm.dev/d", opener=boom)
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


def test_step_ws_upgrade_failed_when_opener_missing() -> None:
    f = network.step_ws_upgrade("wss://farm.dev/d")
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


@pytest.mark.parametrize(
    "p50_ms, expected",
    [
        (10.0, Status.OK),
        (99.999, Status.OK),
        (100.0, Status.DEGRADED),
        (200.0, Status.DEGRADED),
        (300.0, Status.DEGRADED),
        (300.001, Status.FAILED),
        (1000.0, Status.FAILED),
    ],
)
def test_rtt_thresholds_match_design(p50_ms: float, expected: Status) -> None:
    samples = [p50_ms] * 100
    it = iter(samples)
    f = network.step_rtt("farm.dev", samples=100, sampler=lambda h: next(it))
    assert f.status is expected
    assert f.detail["p50_ms"] == pytest.approx(p50_ms)


def test_step_rtt_reports_p50_and_p99() -> None:
    samples = [float(i) for i in range(1, 101)]
    it = iter(samples)
    f = network.step_rtt("farm.dev", samples=100, sampler=lambda h: next(it))
    assert f.detail["samples"] == 100
    assert f.detail["p50_ms"] == pytest.approx(50.5)
    assert f.detail["p99_ms"] == pytest.approx(100.0, rel=0.05)


def test_step_rtt_skips_exceptions_but_passes_when_some_succeed() -> None:
    counter = {"n": 0}

    def sampler(_: str) -> float:
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            raise OSError("blip")
        return 50.0

    f = network.step_rtt("farm.dev", samples=10, sampler=sampler)
    assert f.status is Status.OK
    assert f.detail["samples"] == 5


def test_step_rtt_fails_when_all_samples_raise() -> None:
    def sampler(_: str) -> float:
        raise OSError("down")

    f = network.step_rtt("farm.dev", samples=5, sampler=sampler)
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"
    assert f.detail["samples"] == 0


def test_step_tls_chain_ok_when_verified() -> None:
    f = network.step_tls_chain(
        "farm.dev", checker=lambda h, p: {"verified": True, "cert": {}}
    )
    assert f.status is Status.OK
    assert f.error_code is None


def test_step_tls_chain_failed_when_not_verified() -> None:
    f = network.step_tls_chain(
        "farm.dev", checker=lambda h, p: {"verified": False}
    )
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


def test_step_tls_chain_failed_on_exception() -> None:
    def boom(h: str, p: int) -> dict:
        raise OSError("handshake failed")

    f = network.step_tls_chain("farm.dev", checker=boom)
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


def test_step_mtu_ok_when_above_min() -> None:
    f = network.step_mtu("farm.dev", prober=lambda h, t: 1500)
    assert f.status is Status.OK
    assert f.detail["mtu"] == 1500


def test_step_mtu_degraded_when_low() -> None:
    f = network.step_mtu("farm.dev", prober=lambda h, t: 1000)
    assert f.status is Status.DEGRADED


def test_step_mtu_degraded_on_exception() -> None:
    def boom(h: str, t: int) -> int:
        raise OSError("eperm")

    f = network.step_mtu("farm.dev", prober=boom)
    assert f.status is Status.DEGRADED


def test_step_throughput_ok_at_or_above_25mbps() -> None:
    # 10MB in 3s -> ~26.7 Mbps
    f = network.step_throughput(
        "https://farm.dev/echo",
        size_bytes=10 * 1024 * 1024,
        uploader=lambda url, sz: 3.0,
    )
    assert f.status is Status.OK
    assert f.detail["mbps"] >= 25.0


def test_step_throughput_degraded_between_5_and_25() -> None:
    # 10MB in 10s -> ~8 Mbps
    f = network.step_throughput(
        "https://farm.dev/echo",
        size_bytes=10 * 1024 * 1024,
        uploader=lambda url, sz: 10.0,
    )
    assert f.status is Status.DEGRADED


def test_step_throughput_failed_below_5() -> None:
    # 10MB in 100s -> ~0.8 Mbps
    f = network.step_throughput(
        "https://farm.dev/echo",
        size_bytes=10 * 1024 * 1024,
        uploader=lambda url, sz: 100.0,
    )
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


def test_step_throughput_failed_on_exception() -> None:
    def boom(url: str, sz: int) -> float:
        raise OSError("upload blocked")

    f = network.step_throughput(
        "https://farm.dev/echo", size_bytes=1024, uploader=boom
    )
    assert f.status is Status.FAILED
    assert f.error_code == "FARM-E1007"


def test_verdict_ok_when_all_ok() -> None:
    findings = [
        Finding("a", Status.OK),
        Finding("b", Status.OK),
    ]
    assert network.verdict(findings) is Status.OK


def test_verdict_degraded_when_any_degraded() -> None:
    findings = [
        Finding("a", Status.OK),
        Finding("b", Status.DEGRADED),
    ]
    assert network.verdict(findings) is Status.DEGRADED


def test_verdict_failed_when_any_failed() -> None:
    findings = [
        Finding("a", Status.OK),
        Finding("b", Status.DEGRADED),
        Finding("c", Status.FAILED),
    ]
    assert network.verdict(findings) is Status.FAILED


def test_run_composes_steps_and_writes_verdict() -> None:
    buf = io.StringIO()
    final, findings = network.run(
        out=buf,
        dns_resolver=lambda h: ["192.0.2.1"],
        ws_opener=lambda url: 101,
        rtt_sampler=lambda h: 25.0,
        tls_checker=lambda h, p: {"verified": True, "cert": {}},
        mtu_prober=lambda h, t: 1500,
        throughput_uploader=lambda url, sz: 1.0,
        rtt_samples=10,
    )
    assert final is Status.OK
    assert [f.step for f in findings] == [
        "dns",
        "ws_upgrade",
        "rtt",
        "tls",
        "mtu",
        "throughput",
    ]
    text = buf.getvalue()
    assert "verdict: OK" in text
    for step in ("dns", "ws_upgrade", "rtt", "tls", "mtu", "throughput"):
        assert step in text


def test_run_verdict_failed_when_any_step_fails() -> None:
    buf = io.StringIO()
    final, _ = network.run(
        out=buf,
        dns_resolver=lambda h: ["192.0.2.1"],
        ws_opener=lambda url: 403,
        rtt_sampler=lambda h: 25.0,
        tls_checker=lambda h, p: {"verified": True, "cert": {}},
        mtu_prober=lambda h, t: 1500,
        throughput_uploader=lambda url, sz: 1.0,
        rtt_samples=5,
    )
    assert final is Status.FAILED
    assert "verdict: FAILED" in buf.getvalue()


def test_every_failing_finding_carries_a_fix_code() -> None:
    failing = [
        network.step_dns("farm.dev", resolver=lambda h: []),
        network.step_ws_upgrade("wss://x", opener=lambda u: 500),
        network.step_rtt(samples=1, sampler=lambda h: 999.0),
        network.step_tls_chain(checker=lambda h, p: {"verified": False}),
        network.step_throughput(uploader=lambda u, s: 1000.0),
    ]
    for f in failing:
        assert f.error_code == "FARM-E1007", f.step
        assert f.fix is not None
