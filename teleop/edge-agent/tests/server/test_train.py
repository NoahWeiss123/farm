"""Training-bridge tests — log parser + the /train endpoints' graceful paths.

These never touch the cluster: the no-job status/stop paths and the models
endpoint don't shell out to kubectl, so they stay deterministic.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")  # noqa: F841
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from farm_edge_agent.server import cluster  # noqa: E402
from farm_edge_agent.server.app import build_app  # noqa: E402


@pytest.fixture
async def client():
    app = build_app()
    async with TestClient(TestServer(app)) as c:
        yield c


def test_parse_log_extracts_step_loss_grad():
    text = (
        "starting…\n"
        "Step 0: grad_norm=1.2340, loss=2.5010, param_norm=1800.1\n"
        "NCCL noise line\n"
        "Step 100: grad_norm=0.5340, loss=0.8810, param_norm=1801.2\n"
        "Step 200: grad_norm=0.0334, loss=9.0e-04, param_norm=1806.0\n"
    )
    h = cluster.parse_log(text)
    assert h["steps"] == [0, 100, 200]
    assert h["loss"] == [2.501, 0.881, 9.0e-04]
    assert h["grad_norm"] == [1.234, 0.534, 0.0334]


def test_parse_log_empty_before_training():
    assert cluster.parse_log("installing ffmpeg…\nuv sync…\n") == {
        "steps": [], "loss": [], "grad_norm": []
    }


def test_models_known():
    assert set(cluster.MODELS) == {"full", "lora", "gse"}
    for spec in cluster.MODELS.values():
        assert {"script", "log", "config", "label", "steps", "gpus"} <= set(spec)


async def test_train_models_endpoint(client: TestClient) -> None:
    r = await client.get("/v1/train/models")
    assert r.status == 200
    body = await r.json()
    assert set(body["models"]) == {"full", "lora", "gse"}
    assert body["models"]["gse"]["config"] == "pi05_farm_uf850_gse"
    assert "kubectl" in body


async def test_train_status_no_job(client: TestClient) -> None:
    r = await client.get("/v1/train/status")
    assert r.status == 200
    assert (await r.json())["active"] is False


async def test_train_stop_no_job(client: TestClient) -> None:
    r = await client.post("/v1/train/stop")
    assert r.status == 200
    assert (await r.json())["ok"] is True


async def test_train_launch_unknown_model_rejected(client: TestClient) -> None:
    r = await client.post("/v1/train/launch", json={"model": "nope"})
    assert r.status == 400


async def test_train_page_served(client: TestClient) -> None:
    r = await client.get("/train")
    assert r.status == 200
    assert "Train" in await r.text()
