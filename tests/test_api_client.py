"""app/api_client.py — Streamlit ↔ FastAPI 서빙 API 클라이언트 단위 테스트(이슈 #59 PR 3).

requests 모킹만 사용(실 서버 기동 없음) — 성공·타임아웃·5xx·429·400·base64 복원을 검증한다.
"""
import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from PIL import Image

from api_client import (
    ApiClientError,
    api_base,
    decode_png_base64,
    detect_remote,
    diagnose_remote,
    health_remote,
    prescribe_remote,
)


def _png_base64(color=(255, 0, 0)):
    img = Image.new("RGB", (4, 4), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _diagnosis_response():
    return {
        "ood_blocked": False,
        "reason": None,
        "label": "leaf_mold",
        "label_kr": "잎곰팡이병",
        "prob": 0.91,
        "probs": {"late_blight": 0.02, "leaf_mold": 0.91, "normal": 0.05, "tylcv": 0.02},
        "part": "leaf",
        "cam_png_base64": _png_base64(),
        "yolo": {"annotated_png_base64": _png_base64((0, 255, 0)),
                 "boxes": [{"label": "leaf_mold", "conf": 0.8}]},
    }


# ── api_base() ────────────────────────────────────────────────────────────
def test_api_base_none_when_unset(monkeypatch):
    monkeypatch.delenv("SMARTFARM_API_URL", raising=False)
    assert api_base() is None


def test_api_base_reads_env_and_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000/")
    assert api_base() == "http://127.0.0.1:8000"


def test_api_base_empty_string_is_none(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "")
    assert api_base() is None


# ── diagnose_remote() ────────────────────────────────────────────────────
def test_diagnose_remote_requires_api_base(monkeypatch):
    monkeypatch.delenv("SMARTFARM_API_URL", raising=False)
    with pytest.raises(ApiClientError):
        diagnose_remote(b"fake-bytes")


def test_diagnose_remote_success(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    body = _diagnosis_response()
    with patch("requests.post", return_value=MagicMock(status_code=200, json=lambda: body)) as m:
        r = diagnose_remote(b"fake-bytes", conf=0.4, filename="leaf.jpg")
    assert r == body
    assert m.call_args.args[0] == "http://127.0.0.1:8000/api/diagnoses"
    assert m.call_args.kwargs["params"] == {"conf": 0.4}
    assert m.call_args.kwargs["files"]["file"][0] == "leaf.jpg"
    assert m.call_args.kwargs["timeout"] == 30


def test_diagnose_remote_timeout_raises_connection_failed(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.Timeout("boom")):
        with pytest.raises(ApiClientError, match="서빙 API 연결 실패"):
            diagnose_remote(b"fake-bytes")


def test_diagnose_remote_connection_error_raises_connection_failed(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.ConnectionError("boom")):
        with pytest.raises(ApiClientError, match="서빙 API 연결 실패"):
            diagnose_remote(b"fake-bytes")


def test_diagnose_remote_5xx_raises_connection_failed(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=500, json=lambda: {"detail": "boom"})):
        with pytest.raises(ApiClientError, match="서빙 API 연결 실패"):
            diagnose_remote(b"fake-bytes")


def test_diagnose_remote_429_raises_congestion_message(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=429, json=lambda: {"detail": "congested"})):
        with pytest.raises(ApiClientError, match="혼잡") as exc_info:
            diagnose_remote(b"fake-bytes")
    assert exc_info.value.status_code == 429


def test_diagnose_remote_400_uses_server_detail(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=400, json=lambda: {"detail": "비이미지 파일"})):
        with pytest.raises(ApiClientError, match="비이미지 파일"):
            diagnose_remote(b"fake-bytes")


def test_diagnose_remote_413_uses_server_detail(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=413, json=lambda: {"detail": "10MB 초과"})):
        with pytest.raises(ApiClientError, match="10MB 초과") as exc_info:
            diagnose_remote(b"fake-bytes")
    assert exc_info.value.status_code == 413


def test_diagnose_remote_ood_blocked_passthrough(monkeypatch):
    """게이트 차단도 200 — 클라이언트는 그대로 dict를 반환(뷰가 ood_blocked를 보고 분기)."""
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    blocked = {"ood_blocked": True, "reason": "식물·잎으로 보이지 않는 사진(OOD)", "label": None,
               "label_kr": None, "prob": None, "probs": None, "part": None,
               "cam_png_base64": None, "yolo": None}
    with patch("requests.post", return_value=MagicMock(status_code=200, json=lambda: blocked)):
        r = diagnose_remote(b"fake-bytes")
    assert r["ood_blocked"] is True
    assert r["yolo"] is None


# ── detect_remote() ──────────────────────────────────────────────────────
def _yolo_response():
    return {"annotated_png_base64": _png_base64((0, 255, 0)),
            "boxes": [{"label": "leaf_mold", "conf": 0.8}]}


def test_detect_remote_requires_api_base(monkeypatch):
    monkeypatch.delenv("SMARTFARM_API_URL", raising=False)
    with pytest.raises(ApiClientError):
        detect_remote(b"fake-bytes")


def test_detect_remote_success(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    body = _yolo_response()
    with patch("requests.post", return_value=MagicMock(status_code=200, json=lambda: body)) as m:
        r = detect_remote(b"fake-bytes", conf=0.4, filename="leaf.jpg")
    assert r == body
    assert m.call_args.args[0] == "http://127.0.0.1:8000/api/detections"
    assert m.call_args.kwargs["params"] == {"conf": 0.4}
    assert m.call_args.kwargs["files"]["file"][0] == "leaf.jpg"
    assert m.call_args.kwargs["timeout"] == 30


def test_detect_remote_timeout_raises_connection_failed(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.Timeout("boom")):
        with pytest.raises(ApiClientError, match="서빙 API 연결 실패"):
            detect_remote(b"fake-bytes")


def test_detect_remote_429_raises_congestion_message(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=429, json=lambda: {"detail": "congested"})):
        with pytest.raises(ApiClientError, match="혼잡") as exc_info:
            detect_remote(b"fake-bytes")
    assert exc_info.value.status_code == 429


def test_detect_remote_400_uses_server_detail(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=400, json=lambda: {"detail": "비이미지 파일"})):
        with pytest.raises(ApiClientError, match="비이미지 파일"):
            detect_remote(b"fake-bytes")


# ── prescribe_remote() ───────────────────────────────────────────────────
def test_prescribe_remote_success_with_diagnosis_and_image(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    presc = {"진단요약": "정상입니다", "원인": "-", "즉시조치": "-", "예방": "-",
             "재촬영시점": "-", "근거출처": []}
    diag = {"ood_blocked": False, "label": "normal", "label_kr": "정상", "prob": 0.9,
            "probs": {"normal": 0.9}}
    with patch("requests.post", return_value=MagicMock(status_code=200, json=lambda: presc)) as m:
        r = prescribe_remote("질문", diagnosis=diag, image_bytes=b"img", filename="leaf.jpg")
    assert r == presc
    assert m.call_args.args[0] == "http://127.0.0.1:8000/api/prescriptions"
    assert m.call_args.kwargs["data"]["question"] == "질문"
    assert "leaf_mold" not in m.call_args.kwargs["data"]["diagnosis"]  # sanity: JSON 문자열화됨
    assert m.call_args.kwargs["files"]["file"][0] == "leaf.jpg"
    assert m.call_args.kwargs["timeout"] == 120


def test_prescribe_remote_without_image_omits_files(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    presc = {"진단요약": "-", "원인": "-", "즉시조치": "-", "예방": "-", "재촬영시점": "-", "근거출처": []}
    with patch("requests.post", return_value=MagicMock(status_code=200, json=lambda: presc)) as m:
        prescribe_remote("일반 질문")
    assert m.call_args.kwargs["files"] is None
    assert "diagnosis" not in m.call_args.kwargs["data"]


def test_prescribe_remote_422_raises(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.post", return_value=MagicMock(status_code=422, json=lambda: {"detail": [{"msg": "bad"}]})):
        with pytest.raises(ApiClientError):
            prescribe_remote("질문", diagnosis={"bad": "data"})


# ── health_remote() ──────────────────────────────────────────────────────
def test_health_remote_success(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    body = {"status": "ok", "ollama": {"online": True, "models": ["qwen2.5:14b"]},
            "artifacts": {"resnet18": True, "yolo": True, "part": True, "leaf_gate": True}}
    with patch("requests.get", return_value=MagicMock(status_code=200, json=lambda: body)) as m:
        r = health_remote()
    assert r == body
    assert m.call_args.args[0] == "http://127.0.0.1:8000/api/health"


def test_health_remote_connection_error(monkeypatch):
    monkeypatch.setenv("SMARTFARM_API_URL", "http://127.0.0.1:8000")
    with patch("requests.get", side_effect=requests.ConnectionError("boom")):
        with pytest.raises(ApiClientError, match="서빙 API 연결 실패"):
            health_remote()


# ── decode_png_base64() ──────────────────────────────────────────────────
def test_decode_png_base64_roundtrip():
    b64 = _png_base64((10, 20, 30))
    arr = decode_png_base64(b64)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4, 3)
    assert tuple(arr[0, 0]) == (10, 20, 30)
