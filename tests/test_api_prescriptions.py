"""POST /api/prescriptions — 이슈 #59 PR 2. ollama 모킹(test_prescribe.py 패턴 재사용) + 에러 응답.

Ollama 실호출 없음(로컬 데몬 가동 여부 무관하게 항상 PASS) — 환각 방어·오케스트레이션 자체는
tests/test_prescribe.py가 이미 검증하므로, 여기서는 API 계층(폼 파싱·tempfile·에러 매핑)만 검증한다.
"""
import json
import os
from unittest.mock import patch

from llm import prescribe

_FINAL = json.dumps({
    "진단요약": "잎곰팡이병 의심", "원인": "고온다습", "즉시조치": "감염 잎 제거",
    "예방": "환기", "재촬영시점": "3일 후", "근거출처": [],
}, ensure_ascii=False)
_CHAT_OK = {"message": {"role": "assistant", "content": _FINAL}}


def test_prescriptions_question_only_no_image(api_client, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(prescribe, "get_diagnosis", lambda image_path: calls.__setitem__("n", calls["n"] + 1))
    with patch("ollama.chat", return_value=_CHAT_OK):
        r = api_client.post("/api/prescriptions", data={"question": "오이 병도 알려줘"})
    assert r.status_code == 200
    body = r.json()
    assert body["진단요약"] == "잎곰팡이병 의심"
    assert calls["n"] == 0  # 이미지 없음 → get_diagnosis 자체를 호출하지 않음


def test_prescriptions_reuses_given_diagnosis_skips_get_diagnosis(api_client, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(prescribe, "get_diagnosis", lambda image_path: calls.__setitem__("n", calls["n"] + 1))
    diag = json.dumps({"ood_blocked": False, "label": "leaf_mold", "prob": 0.9, "probs": {}, "part": "leaf"})
    with patch("ollama.chat", return_value=_CHAT_OK):
        r = api_client.post("/api/prescriptions", data={"question": "이 잎 봐줘", "diagnosis": diag})
    assert r.status_code == 200
    assert calls["n"] == 0


def test_prescriptions_with_image_calls_diagnosis_via_tempfile_and_cleans_up(api_client, leaf_image, monkeypatch):
    captured_paths = []

    def _diag(image_path):
        captured_paths.append(image_path)
        assert os.path.exists(image_path)  # 요청 처리 중에는 tempfile이 실재해야 함
        return {"ood_blocked": False, "label": "leaf_mold", "prob": 0.9, "probs": {}, "part": "leaf"}

    monkeypatch.setattr(prescribe, "get_diagnosis", _diag)
    with open(leaf_image, "rb") as f, patch("ollama.chat", return_value=_CHAT_OK):
        r = api_client.post(
            "/api/prescriptions",
            data={"question": "이 잎 봐줘"},
            files={"file": ("leaf.jpg", f, "image/jpeg")},
        )
    assert r.status_code == 200
    assert len(captured_paths) == 1
    assert not os.path.exists(captured_paths[0])  # 응답 후 finally에서 정리됨


def test_prescriptions_invalid_diagnosis_json_is_422(api_client):
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": "not-json"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_prescriptions_empty_file_is_400(api_client):
    r = api_client.post(
        "/api/prescriptions",
        data={"question": "x"},
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_prescriptions_missing_question_is_422(api_client):
    r = api_client.post("/api/prescriptions", data={})
    assert r.status_code == 422


def test_prescriptions_schema_violation_falls_back_gracefully(api_client):
    bad = {"message": {"role": "assistant", "content": "not json"}}
    with patch("ollama.chat", side_effect=[bad, bad]):
        r = api_client.post("/api/prescriptions", data={"question": "오이 병도 알려줘"})
    assert r.status_code == 200
    assert r.json()["진단요약"].startswith("처방 생성에 실패")
