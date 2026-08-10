"""GET /api/health — 이슈 #59 PR 2. Ollama 온/오프라인·아티팩트 존재 여부 모두 200."""
from dl import infer


def test_health_ok_structure(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["ollama"]) == {"online", "models"}
    assert set(body["artifacts"]) == {"resnet18", "yolo", "part", "leaf_gate"}


def test_health_reports_ollama_online_with_models(api_client, monkeypatch):
    """P2-4 — /api/health는 5s 타임아웃 전용 클라이언트(_ollama_health_client())를 쓴다.

    module-level `ollama.list`가 아니라 그 클라이언트 자체를 모킹해야 한다(seam 변경).
    """
    from api.routers import health

    class _Model:
        def __init__(self, name):
            self.model = name

    class _Resp:
        models = [_Model("qwen2.5:14b"), _Model("bge-m3:latest")]

    class _FakeClient:
        def list(self):
            return _Resp()

    monkeypatch.setattr(health, "_ollama_health_client", lambda: _FakeClient())
    r = api_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ollama"] == {"online": True, "models": ["qwen2.5:14b", "bge-m3:latest"]}


def test_health_reports_ollama_offline_without_raising(api_client, monkeypatch):
    """Ollama 데몬 미기동(예외) → 200 + ollama.online=false(서비스 자체는 정상)."""
    from api.routers import health

    class _FakeClient:
        def list(self):
            raise ConnectionError("daemon down")

    monkeypatch.setattr(health, "_ollama_health_client", lambda: _FakeClient())
    r = api_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ollama"] == {"online": False, "models": None}


def test_health_ollama_client_uses_short_timeout():
    """P2-4 — 헬스체크 전용 클라이언트가 5s 타임아웃으로 생성되는지(행 방지)."""
    from api.routers import health

    health._health_ollama_client = None  # 다른 테스트가 이미 캐시했을 수 있음 — 강제 재생성
    client = health._ollama_health_client()
    assert client._client.timeout.connect == health._HEALTH_OLLAMA_TIMEOUT


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
