"""src/llm/prescribe.py — 환각 방어 로직(모킹) + 통합(실 Ollama, 마크)."""
import json
from unittest.mock import patch

import ollama
import pytest

from llm import prescribe
from llm.prescribe import Prescription, _guard_directive

_FINAL = json.dumps({
    "진단요약": "잎곰팡이병 의심", "원인": "고온다습", "즉시조치": "감염 잎 제거",
    "예방": "환기", "재촬영시점": "3일 후", "근거출처": [],
}, ensure_ascii=False)


# ── 환각 방어 ①·② : 신뢰도 밴드·차단·범위밖 지시문 ───────────────────────
def test_guard_high_confidence_asserts():
    assert "단정" in _guard_directive({"ood_blocked": False, "prob": 0.92})


def test_guard_low_confidence_hedges():
    d = _guard_directive({"ood_blocked": False, "prob": 0.3})
    assert "낮다" in d and "단정" in d  # "절대 단정하지 말고"


def test_guard_blocked_asks_retake():
    d = _guard_directive({"ood_blocked": True, "reason": "잎이 아닌 부위"})
    assert "재촬영" in d


def test_guard_none_holds_diagnosis():
    assert "보류" in _guard_directive(None)


def test_guard_error_asks_retry():
    assert "재시도" in _guard_directive({"error": "FileNotFoundError: x"})


# ── 오케스트레이션(ollama 모킹) : 스키마·tool 흐름 ─────────────────────────
def test_prescribe_no_tool_returns_prescription():
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]
    with patch("ollama.chat", side_effect=responses):
        p = prescribe.prescribe("오이 병도 알려줘")  # 범위 밖
    assert isinstance(p, Prescription)
    assert p.진단요약 == "잎곰팡이병 의심"
    assert isinstance(p.근거출처, list)


def test_prescribe_runs_tool_then_validates(monkeypatch):
    """모델이 get_diagnosis 호출 → 실제 tool은 모킹 → 최종 JSON 검증."""
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_diagnosis", "arguments": {"image_path": "x.jpg"}}}]}},
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]
    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis",
                        lambda image_path: {"ood_blocked": False, "label": "leaf_mold",
                                            "prob": 0.9, "probs": {}, "part": "leaf"})
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [])  # RAG 실호출 차단
    with patch("ollama.chat", side_effect=responses):
        p = prescribe.prescribe("이 잎 봐줘", image_path="x.jpg")
    assert isinstance(p, Prescription)
    assert p.즉시조치


def test_prescribe_fills_sources_from_rag(monkeypatch):
    """RAG 검색 결과 → 근거출처가 코드로 채워짐(LLM 아님)."""
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_diagnosis", "arguments": {"image_path": "x.jpg"}}}]}},
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]
    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis",
                        lambda image_path: {"ood_blocked": False, "label": "leaf_mold",
                                            "prob": 0.9, "probs": {}, "part": "leaf"})
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [
        {"title": "잎곰팡이병 방제", "source": "https://ncpms.rda.go.kr/", "text": "환기하라"}])
    with patch("ollama.chat", side_effect=responses):
        p = prescribe.prescribe("이 잎 봐줘", image_path="x.jpg")
    assert p.근거출처 == ["잎곰팡이병 방제 — https://ncpms.rda.go.kr/"]


def test_prescribe_ood_skips_rag(monkeypatch):
    """진단 차단 시 RAG 검색을 아예 호출하지 않는다(엉뚱한 근거 방지)."""
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_diagnosis", "arguments": {"image_path": "x.jpg"}}}]}},
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]
    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis",
                        lambda image_path: {"ood_blocked": True, "reason": "잎 아님"})
    called = {"n": 0}

    def _rt(q, disease=None, k=3):
        called["n"] += 1
        return []

    monkeypatch.setattr(prescribe, "retrieve", _rt)
    with patch("ollama.chat", side_effect=responses):
        p = prescribe.prescribe("봐줘", image_path="x.jpg")
    assert called["n"] == 0
    assert p.근거출처 == []


def test_prescribe_tool_error_still_returns_prescription(monkeypatch):
    """tool 실행 예외 → 잡아서 error 로 되먹이고 여전히 유효한 Prescription 반환(P2.8)."""
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_diagnosis", "arguments": {"image_path": "x.jpg"}}}]}},
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]

    def _boom(image_path):
        raise FileNotFoundError("no such file")

    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis", _boom)
    with patch("ollama.chat", side_effect=responses):
        p = prescribe.prescribe("이 잎 봐줘", image_path="x.jpg")
    assert isinstance(p, Prescription)


def _ollama_up():
    try:
        ollama.list()
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_up(), reason="Ollama 데몬 미실행")
def test_prescribe_real_ollama(leaf_image, monkeypatch, tmp_path):
    # RAG 인덱스 캐시를 tmp 로 격리 — 실 데이터 디렉터리(data/nongsaro) 오염 방지
    from llm.rag import store
    monkeypatch.setattr(store, "INDEX_PATH", tmp_path / "idx.npz")
    p = prescribe.prescribe("이 토마토 잎 진단하고 조치 알려줘", image_path=leaf_image)
    assert isinstance(p, Prescription)
    assert p.진단요약 and p.즉시조치
