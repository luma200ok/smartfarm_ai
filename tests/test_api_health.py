"""GET /api/health — 이슈 #59 PR 2. Ollama 온/오프라인·아티팩트 존재 여부 모두 200."""
from unittest.mock import patch

from dl import infer


def test_health_ok_structure(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["ollama"]) == {"online", "models"}
    assert set(body["artifacts"]) == {"resnet18", "yolo", "part", "leaf_gate"}


def test_health_reports_ollama_online_with_models():
    from fastapi.testclient import TestClient

    from api.main import app

    class _Model:
        def __init__(self, name):
            self.model = name

    class _Resp:
        models = [_Model("qwen2.5:14b"), _Model("bge-m3:latest")]

    with patch("ollama.list", return_value=_Resp()):
        r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json()["ollama"] == {"online": True, "models": ["qwen2.5:14b", "bge-m3:latest"]}


def test_health_reports_ollama_offline_without_raising(api_client):
    """Ollama 데몬 미기동(예외) → 200 + ollama.online=false(서비스 자체는 정상)."""
    with patch("ollama.list", side_effect=ConnectionError("daemon down")):
        r = api_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ollama"] == {"online": False, "models": None}


def test_health_reports_missing_artifacts(api_client, tmp_path, monkeypatch):
    missing = tmp_path / "missing.pt"
    monkeypatch.setattr(infer, "CKPT", missing)
    monkeypatch.setattr(infer, "YOLO_CKPT", missing)
    monkeypatch.setattr(infer, "PART_CKPT", missing)
    r = api_client.get("/api/health")
    assert r.status_code == 200
    artifacts = r.json()["artifacts"]
    assert artifacts["resnet18"] is False
    assert artifacts["yolo"] is False
    assert artifacts["part"] is False
    assert artifacts["leaf_gate"] is True  # 로컬 파일 무관(torchvision 사전학습 가중치)
