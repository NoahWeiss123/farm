"""HTTP API smoke tests — jog, home, gripper, cameras, world stream."""

from __future__ import annotations

import asyncio

import pytest

mujoco = pytest.importorskip("mujoco")  # noqa: F841
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from farm_edge_agent.server.app import build_app  # noqa: E402


@pytest.fixture
async def client():
    app = build_app()
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_healthz_returns_ok(client: TestClient) -> None:
    r = await client.get("/healthz")
    assert r.status == 200
    assert (await r.json())["ok"] is True


async def test_world_has_expected_keys(client: TestClient) -> None:
    r = await client.get("/v1/world")
    assert r.status == 200
    snap = await r.json()
    for key in ("joints", "tcp_pos_mm", "tcp_rpy", "gripper", "t"):
        assert key in snap, f"missing {key}"
    assert len(snap["joints"]) == 6


async def test_dashboard_index_loads(client: TestClient) -> None:
    r = await client.get("/")
    assert r.status == 200
    body = await r.text()
    assert "FARM" in body
    assert "/v1/teleop/jog" in body  # jog endpoint is wired up


async def test_jog_accepts_valid_axis(client: TestClient) -> None:
    r = await client.post("/v1/teleop/jog", json={
        "axis": "z", "sign": 1, "step_mm": 10.0,
    })
    assert r.status == 200, await r.text()
    body = await r.json()
    assert "pose" in body
    assert "snapshot" in body


async def test_jog_rejects_unknown_axis(client: TestClient) -> None:
    r = await client.post("/v1/teleop/jog", json={"axis": "q", "sign": 1})
    assert r.status == 400


async def test_jog_rejects_bad_sign(client: TestClient) -> None:
    r = await client.post("/v1/teleop/jog", json={"axis": "z", "sign": 0})
    assert r.status == 400


async def test_home_endpoint(client: TestClient) -> None:
    r = await client.post("/v1/teleop/home")
    assert r.status == 200
    snap = await r.json()
    assert len(snap["joints"]) == 6


async def test_gripper_endpoint(client: TestClient) -> None:
    r = await client.post("/v1/teleop/gripper", json={"state": "closed"})
    assert r.status == 200
    snap = await r.json()
    assert snap["gripper"] == "closed"
    r2 = await client.post("/v1/teleop/gripper", json={"state": "open"})
    assert r2.status == 200


async def test_camera_jpeg(client: TestClient) -> None:
    r = await client.get("/v1/cameras/exterior.jpg")
    assert r.status == 200
    assert r.headers["Content-Type"] == "image/jpeg"
    body = await r.read()
    assert body[:3] == b"\xff\xd8\xff", "JPEG magic bytes missing"


async def test_camera_unknown_name_returns_404(client: TestClient) -> None:
    r = await client.get("/v1/cameras/nope.jpg")
    assert r.status == 404


async def test_world_stream_emits_at_least_one_snapshot(client: TestClient) -> None:
    r = await client.get("/v1/world/stream", timeout=2.0)
    assert r.status == 200
    # Read enough bytes to get at least one SSE data: line.
    chunk = await asyncio.wait_for(r.content.readline(), timeout=2.0)
    assert chunk.startswith(b"data:"), f"unexpected chunk: {chunk!r}"
    r.release()
