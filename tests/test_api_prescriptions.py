"""POST /api/prescriptions — 이슈 #59 PR 2 + code-reviewer/security-reviewer 후속 픽스.

Ollama 실호출 없음(로컬 데몬 가동 여부 무관하게 항상 PASS) — 환각 방어·오케스트레이션 자체는
tests/test_prescribe.py가 이미 검증하므로, 여기서는 API 계층(폼 파싱·tempfile·에러 매핑·업로드
캡·동시성 캡)만 검증한다.

prescriptions는 fast-path(tool 라운드 없음)라 LLM 호출은 전부 `_write_final`의
`prescribe._client().chat(...)` 1건뿐이다 — `_stub_final_chat`으로만 모킹하면 된다
(P2-4, tests/test_prescribe.py와 동일 seam).
"""
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.concurrency import MAX_CONCURRENT_INFERENCE, _inference_slots
from api.uploads import MAX_UPLOAD_BYTES
from llm import prescribe

_FINAL = json.dumps({
    "진단요약": "잎곰팡이병 의심", "원인": "고온다습", "즉시조치": "감염 잎 제거",
    "예방": "환기", "재촬영시점": "3일 후", "근거출처": [],
}, ensure_ascii=False)
_CHAT_OK = {"message": {"role": "assistant", "content": _FINAL}}


def _stub_final_chat(monkeypatch, *, return_value=None, side_effect=None):
    """`prescribe._client().chat(...)`를 모킹(P2-4 seam) — tests/test_prescribe.py와 동일 패턴."""
    mock_chat = MagicMock(side_effect=side_effect) if side_effect is not None \
        else MagicMock(return_value=return_value)
    monkeypatch.setattr(prescribe, "_client", lambda: SimpleNamespace(chat=mock_chat))
    return mock_chat


def test_prescriptions_question_only_no_image(api_client, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(prescribe, "get_diagnosis", lambda image_path: calls.__setitem__("n", calls["n"] + 1))
    _stub_final_chat(monkeypatch, return_value=_CHAT_OK)
    r = api_client.post("/api/prescriptions", data={"question": "오이 병도 알려줘"})
    assert r.status_code == 200
    body = r.json()
    assert body["진단요약"] == "잎곰팡이병 의심"
    assert calls["n"] == 0  # 이미지 없음 → get_diagnosis 자체를 호출하지 않음


def test_prescriptions_reuses_given_diagnosis_skips_get_diagnosis(api_client, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(prescribe, "get_diagnosis", lambda image_path: calls.__setitem__("n", calls["n"] + 1))
    diag = json.dumps({"ood_blocked": False, "label": "leaf_mold", "prob": 0.9, "probs": {}, "part": "leaf"})
    _stub_final_chat(monkeypatch, return_value=_CHAT_OK)
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
    _stub_final_chat(monkeypatch, return_value=_CHAT_OK)
    with open(leaf_image, "rb") as f:
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
    assert r.json() == {"detail": "diagnosis 필드가 유효한 JSON이 아닙니다."}  # P2-3: 상세 비노출(고정 문구)


def test_prescriptions_empty_file_is_400(api_client):
    r = api_client.post(
        "/api/prescriptions",
        data={"question": "x"},
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_prescriptions_schema_violation_falls_back_gracefully(api_client, monkeypatch):
    bad = {"message": {"role": "assistant", "content": "not json"}}
    mock_chat = _stub_final_chat(monkeypatch, side_effect=[bad, bad])
    r = api_client.post("/api/prescriptions", data={"question": "오이 병도 알려줘"})
    assert r.status_code == 200
    assert r.json()["진단요약"].startswith("처방 생성에 실패")
    assert mock_chat.call_count == 2


def test_prescriptions_missing_question_is_422(api_client):
    r = api_client.post("/api/prescriptions", data={})
    assert r.status_code == 422


# ── P1-1: 업로드 크기 캡 ────────────────────────────────────────────────
def test_prescriptions_oversized_file_is_413(api_client):
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    r = api_client.post(
        "/api/prescriptions",
        data={"question": "x"},
        files={"file": ("big.jpg", oversized, "image/jpeg")},
    )
    assert r.status_code == 413
    assert r.json() == {"detail": "파일이 너무 큽니다(최대 10MB)."}


# ── P2-1: diagnosis Form 검증(비-dict 500 재현 케이스 고정 + 화이트리스트) ────
def test_prescriptions_diagnosis_non_dict_int_is_422_not_500(api_client):
    """리뷰 재현 케이스 — diagnosis="5"(정수) 입력 시 과거엔 diag.get() 호출에서 500이 났다."""
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": "5"})
    assert r.status_code == 422
    assert r.json() == {"detail": "diagnosis 필드 형식이 올바르지 않습니다."}


def test_prescriptions_diagnosis_non_dict_list_is_422_not_500(api_client):
    """리뷰 재현 케이스 — diagnosis="[1,2,3]"(배열) 입력 시 과거엔 500이 났다."""
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": "[1,2,3]"})
    assert r.status_code == 422


def test_prescriptions_diagnosis_unknown_label_is_422(api_client):
    diag = json.dumps({"ood_blocked": False, "label": "not_a_real_class", "prob": 0.9})
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": diag})
    assert r.status_code == 422


def test_prescriptions_diagnosis_prob_out_of_range_is_422(api_client):
    diag = json.dumps({"ood_blocked": False, "label": "leaf_mold", "prob": 1.5})
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": diag})
    assert r.status_code == 422


def test_prescriptions_diagnosis_probs_value_out_of_range_is_422(api_client):
    diag = json.dumps({"ood_blocked": False, "label": "leaf_mold", "probs": {"leaf_mold": 1.2}})
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": diag})
    assert r.status_code == 422


def test_prescriptions_diagnosis_label_none_is_allowed(api_client, monkeypatch):
    """label=None(예: ood_blocked 케이스)은 화이트리스트 검사 대상이 아니다 — 정상 통과."""
    diag = json.dumps({"ood_blocked": True, "reason": "잎이 아닌 부위", "label": None})
    _stub_final_chat(monkeypatch, return_value=_CHAT_OK)
    r = api_client.post("/api/prescriptions", data={"question": "x", "diagnosis": diag})
    assert r.status_code == 200


# ── P2-2: 동시성 캡(429) ───────────────────────────────────────────────
def test_prescriptions_returns_429_when_slots_exhausted(api_client):
    """diagnoses·prescriptions가 슬롯(전역 세마포어)을 공유하므로, 슬롯을 미리 다 채우면 429."""
    acquired = [_inference_slots.acquire(blocking=False) for _ in range(MAX_CONCURRENT_INFERENCE)]
    assert all(acquired)
    try:
        r = api_client.post("/api/prescriptions", data={"question": "x"})
        assert r.status_code == 429
        assert r.json() == {"detail": "서버가 혼잡합니다. 잠시 후 다시 시도하세요."}
    finally:
        for _ in acquired:
            _inference_slots.release()


# ── P3-1: 비이미지 업로드는 진단·LLM 호출 전에 조기 차단 ────────────────────
def test_prescriptions_non_image_file_is_400_before_diagnosis_or_llm(api_client, monkeypatch):
    diag_calls = {"n": 0}
    monkeypatch.setattr(prescribe, "get_diagnosis", lambda image_path: diag_calls.__setitem__("n", diag_calls["n"] + 1))
    mock_chat = _stub_final_chat(monkeypatch, return_value=_CHAT_OK)
    r = api_client.post(
        "/api/prescriptions",
        data={"question": "x"},
        files={"file": ("x.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400
    assert diag_calls["n"] == 0  # 진단까지 흘러가지 않음
    mock_chat.assert_not_called()  # LLM 호출까지 흘러가지 않음
