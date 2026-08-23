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

from api.concurrency import MAX_CONCURRENT_CHAT, MAX_CONCURRENT_INFERENCE, _chat_slots, _inference_slots
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


# ── 챗 전용 하위 상한(security-reviewer P2, smartfarm_ai#84) ─────────────
# 챗이 공유 inference_slot(합계 2)을 전부 점유해 진단·처방을 굶기지 못하도록,
# chat_slot(MAX_CONCURRENT_CHAT=1)을 먼저 획득해야만 inference_slot을 시도한다.
def test_chat_second_concurrent_chat_is_429_even_with_shared_slots_free(api_client):
    """chat_slot을 1개 미리 점유하면(=챗 1건 처리 중) 공유 inference_slot이 남아 있어도
    두 번째 챗 요청은 429여야 한다(챗은 최대 1슬롯만 쓸 수 있음)."""
    assert _chat_slots.acquire(blocking=False) is True
    try:
        assert _inference_slots._value == MAX_CONCURRENT_INFERENCE  # 공유 슬롯은 아직 안 건드림
        r = api_client.post("/api/chat", data={"question": "질문"})
        assert r.status_code == 429
        assert r.json() == {"detail": "서버가 혼잡합니다. 잠시 후 다시 시도하세요."}
    finally:
        _chat_slots.release()


def test_chat_holds_at_most_one_shared_slot_leaving_room_for_diagnosis_prescription(api_client, monkeypatch):
    """챗 처리 도중(핸들러 안에서) 공유 슬롯은 정확히 1개만 소비돼야 한다 —
    진단·처방용으로 최소 1개(MAX_CONCURRENT_INFERENCE - 1)가 항상 남아 있어야 한다."""
    monkeypatch.setattr(chat, "retrieve", lambda q, disease=None, k=3: [])
    observed = {}

    def _fake_chat(*args, **kwargs):
        # LLM 호출 시점 = 챗이 chat_slot·inference_slot을 모두 쥐고 있는 상태
        observed["inference_free"] = _inference_slots._value
        observed["chat_free"] = _chat_slots._value
        return {"message": {"role": "assistant", "content": "답변"}}

    monkeypatch.setattr(chat, "_client", lambda: SimpleNamespace(chat=_fake_chat))
    r = api_client.post("/api/chat", data={"question": "질문"})
    assert r.status_code == 200
    assert observed["inference_free"] == MAX_CONCURRENT_INFERENCE - 1  # 진단·처방용 1개 이상 남음
    assert observed["chat_free"] == MAX_CONCURRENT_CHAT - 1
    # 요청 종료 후엔 둘 다 원복(release 확인)
    assert _inference_slots._value == MAX_CONCURRENT_INFERENCE
    assert _chat_slots._value == MAX_CONCURRENT_CHAT


def test_chat_exhausted_shared_inference_slots_still_429_after_chat_slot_acquired(api_client):
    """chat_slot은 남아 있어도(챗 자기 상한은 안 참) 공유 inference_slot이 다 차 있으면 429."""
    acquired = [_inference_slots.acquire(blocking=False) for _ in range(MAX_CONCURRENT_INFERENCE)]
    assert all(acquired)
    try:
        r = api_client.post("/api/chat", data={"question": "질문"})
        assert r.status_code == 429
        # chat_slot은 정상적으로 획득했다가 inference_slot 실패로 해제돼 원복돼 있어야 함
        assert _chat_slots._value == MAX_CONCURRENT_CHAT
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
