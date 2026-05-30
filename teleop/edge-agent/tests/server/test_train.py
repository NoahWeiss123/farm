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


def test_parse_log_step_order_independent():
    # Key order varies by config — loss may come before grad_norm.
    h = cluster.parse_log("Step 50: loss=0.4200, grad_norm=0.1100, param_norm=12.0\n")
    assert h["steps"] == [50] and h["loss"] == [0.42] and h["grad_norm"] == [0.11]


def test_parse_log_tqdm_progress():
    # openpi's tqdm→logging line — the reliable live-progress signal.
    text = (
        "Progress on: 326it/3.00kit rate:1.4s/it remaining:1:04:17 elapsed:09:13 postfix:- "
        "(548075:tqdm_logging.py:145)\n"
        "Progress on: 378it/3.00kit rate:1.5s/it remaining:1:05:10 elapsed:10:30 postfix:- "
        "(548075:tqdm_logging.py:145)\n"
    )
    p = cluster.parse_log(text)["progress"]
    assert p["step"] == 378 and p["total"] == 3000          # last line wins
    assert p["s_per_it"] == 1.5 and p["it_per_s"] == round(1 / 1.5, 3)
    assert p["remaining_s"] == 1 * 3600 + 5 * 60 + 10        # 1:05:10
    assert p["elapsed_s"] == 10 * 60 + 30                    # 10:30


def test_parse_log_empty_before_training():
    assert cluster.parse_log("installing ffmpeg…\nuv sync…\n") == {
        "steps": [], "loss": [], "grad_norm": [], "progress": None
    }


def test_model_from_name():
    assert cluster._model_from_name("farm-pi05-gse") == "gse"
    assert cluster._model_from_name("farm-pi05-lora") == "lora"
    assert cluster._model_from_name("farm-pi05") == "full"
    assert cluster._model_from_name("serve-pi05") is None


def test_parse_metrics_gpu_and_cpu():
    blob = "0, 95, 41000, 81920\n1, 88, 40000, 81920\nCPU\n1.50 1.20 0.90 2/300 999\n16\n"
    m = cluster.parse_metrics(blob)
    assert len(m["gpus"]) == 2
    assert m["gpus"][0] == {"index": 0, "util": 95, "mem_used": 41000, "mem_total": 81920, "mem_pct": 50.0}
    assert m["gpus"][1]["util"] == 88
    assert m["cpu"]["load1"] == 1.5 and m["cpu"]["ncpu"] == 16
    assert m["cpu"]["pct"] == round(100 * 1.5 / 16, 1)


def test_parse_metrics_empty():
    assert cluster.parse_metrics("") == {"gpus": [], "cpu": {}}


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


async def test_train_metrics_no_job(client: TestClient) -> None:
    r = await client.get("/v1/train/metrics")
    assert r.status == 200
    assert (await r.json())["active"] is False


async def test_train_launch_unknown_model_rejected(client: TestClient) -> None:
    r = await client.post("/v1/train/launch", json={"model": "nope"})
    assert r.status == 400


async def test_train_page_served(client: TestClient) -> None:
    r = await client.get("/train")
    assert r.status == 200
    assert "Train" in await r.text()
