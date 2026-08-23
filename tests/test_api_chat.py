"""POST /api/chat — smartfarm_ai#84. RAG 자유 질의 + 이력.

실 Ollama·pgvector 호출 없음(로컬 데몬/DB 가동 여부 무관하게 항상 PASS) — 환각 방어·프롬프트
구성 자체는 이 파일에서 `chat._client().chat(...)` 1건만 모킹해 검증하고(tests/test_api_prescriptions.py
와 동일 seam 원칙), RAG 검색도 `chat.retrieve`를 스텁해 코퍼스/임베딩 의존을 없앤다.

빈 question(Form 빈 문자열)은 FastAPI가 필수 폼 필드의 빈 값을 "미제공"으로 취급해 항상
`{"detail":[{"type":"missing", ...}]}` 형태로 422를 반환한다(min_length 위반이 아니라 필드
누락과 동일 경로) — 두 케이스 모두 422만 확인한다.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.concurrency import MAX_CONCURRENT_INFERENCE, _inference_slots
from llm import chat

_RAG_CHUNKS = [
    {"title": "잎곰팡이병 방제", "text": "환기와 습도 관리가 중요하다.",
     "source": "https://www.nongsaro.go.kr/", "source_name": "농촌진흥청", "disease": "leaf_mold"},
]


def _stub_chat_client(monkeypatch, *, return_value=None, side_effect=None):
    """`chat._client().chat(...)`을 모킹(prescribe.py `_write_final`과 동일 seam 원칙)."""
    mock_chat = MagicMock(side_effect=side_effect) if side_effect is not None \
        else MagicMock(return_value=return_value)
    monkeypatch.setattr(chat, "_client", lambda: SimpleNamespace(chat=mock_chat))
    return mock_chat


# ── 정상 경로 ────────────────────────────────────────────────────────────
def test_chat_normal_returns_answer_sources_and_fallback_false(api_client, monkeypatch):
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: _RAG_CHUNKS)
    _stub_chat_client(monkeypatch, return_value={
        "message": {"role": "assistant", "content": "환기를 자주 해주세요."},
    })
    r = api_client.post("/api/chat", data={"question": "잎곰팡이병 예방법이 뭐야?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "환기를 자주 해주세요."
    assert body["sources"] == ["잎곰팡이병 방제 (농촌진흥청) — https://www.nongsaro.go.kr/"]
    assert body["fallback"] is False


def test_chat_no_rag_hits_returns_empty_sources(api_client, monkeypatch):
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    _stub_chat_client(monkeypatch, return_value={
        "message": {"role": "assistant", "content": "토마토 재배 질문을 도와드릴게요."},
    })
    r = api_client.post("/api/chat", data={"question": "오늘 날씨 어때?"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


# ── 검증 실패(422) ───────────────────────────────────────────────────────
def test_chat_missing_question_is_422(api_client):
    r = api_client.post("/api/chat", data={})
    assert r.status_code == 422


def test_chat_empty_question_is_422(api_client):
    r = api_client.post("/api/chat", data={"question": ""})
    assert r.status_code == 422


def test_chat_question_over_500_chars_is_422(api_client):
    r = api_client.post("/api/chat", data={"question": "가" * 501})
    assert r.status_code == 422


def test_chat_caller_ref_over_64_chars_is_422(api_client):
    r = api_client.post("/api/chat", data={"question": "질문", "caller_ref": "a" * 65})
    assert r.status_code == 422


# ── 동시성 캡(429, diagnoses·prescriptions와 슬롯 공유) ───────────────────
def test_chat_returns_429_when_slots_exhausted(api_client):
    acquired = [_inference_slots.acquire(blocking=False) for _ in range(MAX_CONCURRENT_INFERENCE)]
    assert all(acquired)
    try:
        r = api_client.post("/api/chat", data={"question": "질문"})
        assert r.status_code == 429
        assert r.json() == {"detail": "서버가 혼잡합니다. 잠시 후 다시 시도하세요."}
    finally:
        for _ in acquired:
            _inference_slots.release()


# ── LLM 실패 → 안전 폴백(200 + fallback=true) ─────────────────────────────
def test_chat_llm_exception_falls_back_gracefully(api_client, monkeypatch):
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    mock_chat = _stub_chat_client(monkeypatch, side_effect=RuntimeError("ollama 연결 실패"))
    r = api_client.post("/api/chat", data={"question": "질문"})
    assert r.status_code == 200
    body = r.json()
    assert body["fallback"] is True
    assert body["answer"]  # 빈 문자열이 아닌 한국어 안내문
    mock_chat.assert_called_once()


def test_chat_llm_empty_response_falls_back_gracefully(api_client, monkeypatch):
    """빈 문자열 응답도 스키마 위반은 아니지만 무의미한 답이라 폴백 처리한다."""
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    _stub_chat_client(monkeypatch, return_value={"message": {"role": "assistant", "content": "   "}})
    r = api_client.post("/api/chat", data={"question": "질문"})
    assert r.status_code == 200
    assert r.json()["fallback"] is True


# ── 이력 저장(best-effort) ────────────────────────────────────────────────
def test_chat_saves_history_with_caller_ref(api_client, monkeypatch):
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    _stub_chat_client(monkeypatch, return_value={"message": {"role": "assistant", "content": "답변입니다."}})
    captured = {}
    monkeypatch.setattr(
        chat.history, "save_chat",
        lambda question, answer, sources, caller_ref=None: captured.update(
            question=question, answer=answer, sources=sources, caller_ref=caller_ref),
    )
    r = api_client.post("/api/chat", data={"question": "질문입니다", "caller_ref": "tenant-x"})
    assert r.status_code == 200
    assert captured["question"] == "질문입니다"
    assert captured["answer"] == "답변입니다."
    assert captured["sources"] == []
    assert captured["caller_ref"] == "tenant-x"


def test_chat_caller_ref_omitted_defaults_none_in_history(api_client, monkeypatch):
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    _stub_chat_client(monkeypatch, return_value={"message": {"role": "assistant", "content": "답변"}})
    captured = {}
    monkeypatch.setattr(
        chat.history, "save_chat",
        lambda question, answer, sources, caller_ref=None: captured.update(caller_ref=caller_ref),
    )
    r = api_client.post("/api/chat", data={"question": "질문"})
    assert r.status_code == 200
    assert captured["caller_ref"] is None


def test_chat_saves_history_even_on_llm_fallback(api_client, monkeypatch):
    """LLM 실패로 폴백을 반환해도 이력 저장은 시도된다(best-effort, 응답을 막지 않음)."""
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    _stub_chat_client(monkeypatch, side_effect=RuntimeError("ollama 연결 실패"))
    captured = {}
    monkeypatch.setattr(
        chat.history, "save_chat",
        lambda question, answer, sources, caller_ref=None: captured.update(answer=answer),
    )
    r = api_client.post("/api/chat", data={"question": "질문"})
    assert r.status_code == 200
    assert captured["answer"] == r.json()["answer"]
