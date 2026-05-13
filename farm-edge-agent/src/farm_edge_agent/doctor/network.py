from __future__ import annotations

import socket
import ssl
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import IO, Any, Callable


class Status(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Finding:
    step: str
    status: Status
    detail: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    fix: str | None = None


RTT_OK_MS = 100.0
RTT_DEGRADED_MAX_MS = 300.0
THROUGHPUT_OK_MBPS = 25.0
THROUGHPUT_DEGRADED_MIN_MBPS = 5.0
MTU_OK_MIN = 1280
DEFAULT_HOST = "farm.dev"
DEFAULT_WS_URL = "wss://farm.dev/dispatcher"
DEFAULT_ECHO_URL = "https://farm.dev/echo"
DEFAULT_RTT_SAMPLES = 100
DEFAULT_UPLOAD_BYTES = 10 * 1024 * 1024


def step_dns(
    host: str = DEFAULT_HOST,
    resolver: Callable[[str], list[str]] | None = None,
) -> Finding:
    _resolve = resolver or _default_dns_resolver
    try:
        addrs = _resolve(host)
    except Exception as exc:
        return Finding(
            step="dns",
            status=Status.FAILED,
            detail={"host": host, "error": str(exc)},
            error_code="FARM-E1007",
            fix="DNS resolution failed; try `nslookup farm.dev` or switch network.",
        )
    if not addrs:
        return Finding(
            step="dns",
            status=Status.FAILED,
            detail={"host": host, "addrs": []},
            error_code="FARM-E1007",
            fix="DNS resolver returned no addresses; switch network.",
        )
    return Finding(
        step="dns",
        status=Status.OK,
        detail={"host": host, "addrs": list(addrs)},
    )


def step_ws_upgrade(
    url: str = DEFAULT_WS_URL,
    opener: Callable[[str], int] | None = None,
) -> Finding:
    if opener is None:
        return Finding(
            step="ws_upgrade",
            status=Status.FAILED,
            detail={"url": url, "error": "no opener provided"},
            error_code="FARM-E1007",
            fix="WebSocket upgrade not attempted; try FARM_RELAY=on.",
        )
    try:
        code = opener(url)
    except Exception as exc:
        return Finding(
            step="ws_upgrade",
            status=Status.FAILED,
            detail={"url": url, "error": str(exc)},
            error_code="FARM-E1007",
            fix="WebSocket upgrade blocked; try FARM_RELAY=on.",
        )
    if code == 101:
        return Finding(
            step="ws_upgrade",
            status=Status.OK,
            detail={"url": url, "status": code},
        )
    return Finding(
        step="ws_upgrade",
        status=Status.FAILED,
        detail={"url": url, "status": code},
        error_code="FARM-E1007",
        fix="WebSocket upgrade rejected; try FARM_RELAY=on.",
    )


def _classify_rtt(p50_ms: float) -> Status:
    if p50_ms < RTT_OK_MS:
        return Status.OK
    if p50_ms <= RTT_DEGRADED_MAX_MS:
        return Status.DEGRADED
    return Status.FAILED


def step_rtt(
    host: str = DEFAULT_HOST,
    samples: int = DEFAULT_RTT_SAMPLES,
    sampler: Callable[[str], float] | None = None,
) -> Finding:
    if sampler is None:
        return Finding(
            step="rtt",
            status=Status.FAILED,
            detail={"host": host, "error": "no sampler provided"},
            error_code="FARM-E1007",
            fix="RTT probe needs a working network; run on the target machine.",
        )
    rtts: list[float] = []
    for _ in range(samples):
        try:
            rtts.append(sampler(host))
        except Exception:
            continue
    if not rtts:
        return Finding(
            step="rtt",
            status=Status.FAILED,
            detail={"host": host, "samples": 0},
            error_code="FARM-E1007",
            fix="No RTT samples collected; switch network.",
        )
    p50 = statistics.median(rtts)
    p99 = _percentile(rtts, 99)
    status = _classify_rtt(p50)
    detail = {
        "host": host,
        "samples": len(rtts),
        "p50_ms": p50,
        "p99_ms": p99,
    }
    if status is Status.OK:
        return Finding(step="rtt", status=status, detail=detail)
    return Finding(
        step="rtt",
        status=status,
        detail=detail,
        error_code="FARM-E1007",
        fix="High RTT to dispatcher; use a wired link or closer region.",
    )


def step_tls_chain(
    host: str = DEFAULT_HOST,
    port: int = 443,
    checker: Callable[[str, int], dict[str, Any]] | None = None,
) -> Finding:
    if checker is None:
        return Finding(
            step="tls",
            status=Status.FAILED,
            detail={"host": host, "error": "no checker provided"},
            error_code="FARM-E1007",
            fix="TLS check skipped; verify network reachability.",
        )
    try:
        result = checker(host, port)
    except Exception as exc:
        return Finding(
            step="tls",
            status=Status.FAILED,
            detail={"host": host, "error": str(exc)},
            error_code="FARM-E1007",
            fix="TLS handshake failed; check system CA store.",
        )
    if not result.get("verified", False):
        return Finding(
            step="tls",
            status=Status.FAILED,
            detail={"host": host, "cert": result},
            error_code="FARM-E1007",
            fix="TLS chain not trusted; update CA bundle.",
        )
    return Finding(
        step="tls",
        status=Status.OK,
        detail={"host": host, "cert": result},
    )


def step_mtu(
    host: str = DEFAULT_HOST,
    target: int = 1500,
    prober: Callable[[str, int], int] | None = None,
) -> Finding:
    if prober is None:
        return Finding(
            step="mtu",
            status=Status.DEGRADED,
            detail={"host": host, "error": "no prober provided"},
            fix="MTU probe skipped; defaulting to DEGRADED.",
        )
    try:
        mtu = prober(host, target)
    except Exception as exc:
        return Finding(
            step="mtu",
            status=Status.DEGRADED,
            detail={"host": host, "error": str(exc)},
            fix="MTU probe failed; expect possible fragmentation.",
        )
    if mtu >= MTU_OK_MIN:
        return Finding(
            step="mtu",
            status=Status.OK,
            detail={"host": host, "mtu": mtu},
        )
    return Finding(
        step="mtu",
        status=Status.DEGRADED,
        detail={"host": host, "mtu": mtu},
        fix="Path MTU is small; large action chunks may fragment.",
    )


def step_throughput(
    url: str = DEFAULT_ECHO_URL,
    size_bytes: int = DEFAULT_UPLOAD_BYTES,
    uploader: Callable[[str, int], float] | None = None,
) -> Finding:
    if uploader is None:
        return Finding(
            step="throughput",
            status=Status.FAILED,
            detail={"url": url, "error": "no uploader provided"},
            error_code="FARM-E1007",
            fix="Throughput probe needs network; try FARM_RELAY=on.",
        )
    try:
        elapsed = uploader(url, size_bytes)
    except Exception as exc:
        return Finding(
            step="throughput",
            status=Status.FAILED,
            detail={"url": url, "error": str(exc)},
            error_code="FARM-E1007",
            fix="Upload failed; try FARM_RELAY=on.",
        )
    if elapsed <= 0:
        return Finding(
            step="throughput",
            status=Status.FAILED,
            detail={"url": url, "elapsed_s": elapsed},
            error_code="FARM-E1007",
            fix="Throughput probe reported non-positive elapsed time.",
        )
    mbps = (size_bytes * 8.0) / (elapsed * 1_000_000)
    detail = {"url": url, "elapsed_s": elapsed, "mbps": mbps, "bytes": size_bytes}
    if mbps >= THROUGHPUT_OK_MBPS:
        return Finding(step="throughput", status=Status.OK, detail=detail)
    if mbps >= THROUGHPUT_DEGRADED_MIN_MBPS:
        return Finding(
            step="throughput",
            status=Status.DEGRADED,
            detail=detail,
            fix="Throughput below 25Mbps; expect higher run latency.",
        )
    return Finding(
        step="throughput",
        status=Status.FAILED,
        detail=detail,
        error_code="FARM-E1007",
        fix="Throughput too low; switch network or try FARM_RELAY=on.",
    )


def verdict(findings: Sequence[Finding]) -> Status:
    if any(f.status is Status.FAILED for f in findings):
        return Status.FAILED
    if any(f.status is Status.DEGRADED for f in findings):
        return Status.DEGRADED
    return Status.OK


def format_finding(f: Finding) -> str:
    parts = [f"{f.step:11s} {f.status.value:8s}"]
    if f.error_code is not None:
        parts.append(f"[{f.error_code}]")
    if f.detail:
        parts.append(_compact_detail(f.detail))
    if f.fix:
        parts.append(f"fix: {f.fix}")
    return "  ".join(parts)


def _compact_detail(detail: dict[str, Any]) -> str:
    pieces = []
    for key, value in detail.items():
        if isinstance(value, float):
            pieces.append(f"{key}={value:.2f}")
        elif isinstance(value, list):
            pieces.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            pieces.append(f"{key}={value}")
    return " ".join(pieces)


def run(
    out: IO[str] | None = None,
    dns_resolver: Callable[[str], list[str]] | None = None,
    ws_opener: Callable[[str], int] | None = None,
    rtt_sampler: Callable[[str], float] | None = None,
    tls_checker: Callable[[str, int], dict[str, Any]] | None = None,
    mtu_prober: Callable[[str, int], int] | None = None,
    throughput_uploader: Callable[[str, int], float] | None = None,
    rtt_samples: int = DEFAULT_RTT_SAMPLES,
) -> tuple[Status, list[Finding]]:
    stream = out if out is not None else sys.stdout
    findings: list[Finding] = [
        step_dns(resolver=dns_resolver or _default_dns_resolver),
        step_ws_upgrade(opener=ws_opener),
        step_rtt(samples=rtt_samples, sampler=rtt_sampler or _default_rtt_sampler),
        step_tls_chain(checker=tls_checker or _default_tls_checker),
        step_mtu(prober=mtu_prober or _default_mtu_prober),
        step_throughput(uploader=throughput_uploader),
    ]
    final = verdict(findings)
    for f in findings:
        stream.write(format_finding(f) + "\n")
    stream.write(f"verdict: {final.value}\n")
    return final, findings


def _default_dns_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return sorted({info[4][0] for info in infos})


def _default_rtt_sampler(host: str) -> float:
    family, socktype, proto, _, sockaddr = socket.getaddrinfo(
        host, 443, type=socket.SOCK_STREAM
    )[0]
    sock = socket.socket(family, socktype, proto)
    sock.settimeout(5)
    start = time.perf_counter()
    try:
        sock.connect(sockaddr)
    finally:
        sock.close()
    return (time.perf_counter() - start) * 1000.0


def _default_tls_checker(host: str, port: int) -> dict[str, Any]:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=5) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as tls:
            cert = tls.getpeercert() or {}
            return {"verified": True, "cert": cert}


def _default_mtu_prober(host: str, target: int) -> int:
    del host
    return target


def _percentile(samples: list[float], pct: int) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
