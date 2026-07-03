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
    monkeypatch.setattr(prescribe, "get_forecast", lambda: {"unavailable": True})  # forecast 실호출 차단
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
    monkeypatch.setattr(prescribe, "get_forecast", lambda: {"unavailable": True})
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


def _diag_flow_responses():
    return [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_diagnosis", "arguments": {"image_path": "x.jpg"}}}]}},
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]


def test_prescribe_injects_forecast_for_leaf_mold(monkeypatch):
    """습도 민감 병해(leaf_mold) → 시간축 처방 위해 get_forecast 호출."""
    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis",
                        lambda image_path: {"ood_blocked": False, "label": "leaf_mold",
                                            "prob": 0.9, "probs": {}, "part": "leaf"})
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [])
    called = {"n": 0}

    def _fc():
        called["n"] += 1
        return {"next_temp": 30.0, "trend": "유지", "humidity_risk": "높음",
                "humidity_mean": 90.0, "recent_temp": 29.0}

    monkeypatch.setattr(prescribe, "get_forecast", _fc)
    with patch("ollama.chat", side_effect=_diag_flow_responses()):
        prescribe.prescribe("이 잎 봐줘", image_path="x.jpg")
    assert called["n"] == 1


def test_prescribe_skips_forecast_for_non_leafmold(monkeypatch):
    """tylcv 등 비-곰팡이 진단에는 forecast 교차를 넣지 않는다."""
    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis",
                        lambda image_path: {"ood_blocked": False, "label": "tylcv",
                                            "prob": 0.9, "probs": {}, "part": "leaf"})
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [])
    called = {"n": 0}
    monkeypatch.setattr(prescribe, "get_forecast",
                        lambda: called.__setitem__("n", called["n"] + 1) or {"unavailable": True})
    with patch("ollama.chat", side_effect=_diag_flow_responses()):
        prescribe.prescribe("이 잎 봐줘", image_path="x.jpg")
    assert called["n"] == 0


def test_prescribe_tool_round_caps_num_predict_and_sets_keep_alive():
    """C1(#15) — tool 라운드 chat 호출에 num_predict 캡·keep_alive 전달 검증(낭비 라운드 지연 완화)."""
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]
    with patch("ollama.chat", side_effect=responses) as mock_chat:
        prescribe.prescribe("오이 병도 알려줘")
    first_call_kwargs = mock_chat.call_args_list[0].kwargs
    assert first_call_kwargs["options"] == {"num_predict": prescribe.TOOL_ROUND_NUM_PREDICT}
    assert first_call_kwargs["keep_alive"] == prescribe.KEEP_ALIVE
    final_call_kwargs = mock_chat.call_args_list[1].kwargs
    assert final_call_kwargs["keep_alive"] == prescribe.KEEP_ALIVE


def test_prescribe_warns_when_diagnosis_not_called_with_image(monkeypatch, caplog):
    """P1 — 이미지 첨부됐는데 tool 미호출로 종료되면 경고 로그로 가시화(조용한 '진단 보류' 방지)."""
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]
    with caplog.at_level("WARNING"):
        with patch("ollama.chat", side_effect=responses):
            prescribe.prescribe("이 잎 봐줘", image_path="x.jpg")
    assert any("tool 호출 없이 종료" in r.message for r in caplog.records)


def test_prescribe_on_progress_exception_does_not_break_prescription():
    """P2-1 — on_progress 콜백이 raise해도 처방은 정상 반환된다."""
    def _boom(stage):
        raise RuntimeError("ui 콜백 실패")

    with patch("ollama.chat", side_effect=[
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]):
        p = prescribe.prescribe("오이 병도 알려줘", on_progress=_boom)
    assert isinstance(p, Prescription)


def test_prescribe_on_progress_reports_stages(monkeypatch):
    """C3(#15) — on_progress 콜백이 diagnosis→rag→forecast→writing 단계 순서로 호출된다."""
    monkeypatch.setitem(prescribe.TOOL_REGISTRY, "get_diagnosis",
                        lambda image_path: {"ood_blocked": False, "label": "leaf_mold",
                                            "prob": 0.9, "probs": {}, "part": "leaf"})
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [
        {"title": "잎곰팡이병 방제", "source": "https://ncpms.rda.go.kr/", "text": "환기하라"}])
    monkeypatch.setattr(prescribe, "get_forecast",
                        lambda: {"next_temp": 30.0, "trend": "유지", "humidity_risk": "높음",
                                "humidity_mean": 90.0, "recent_temp": 29.0})
    stages: list[str] = []
    with patch("ollama.chat", side_effect=_diag_flow_responses()):
        prescribe.prescribe("이 잎 봐줘", image_path="x.jpg", on_progress=stages.append)
    assert stages == ["diagnosis", "rag", "forecast", "writing"]


def test_prescribe_no_on_progress_unchanged():
    """on_progress 미지정(None)이면 예외 없이 기존과 동일하게 동작(하위호환)."""
    with patch("ollama.chat", side_effect=[
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": _FINAL}},
    ]):
        p = prescribe.prescribe("오이 병도 알려줘")
    assert isinstance(p, Prescription)


# ── fast-path(1-call, 이슈 #18) ────────────────────────────────────────
def test_fast_calls_diagnosis_directly_not_via_llm(monkeypatch):
    """이미지 있으면 코드가 직접 get_diagnosis를 호출 — LLM tool 선택 없이."""
    calls = {"n": 0}

    def _diag(image_path):
        calls["n"] += 1
        return {"ood_blocked": False, "label": "leaf_mold", "prob": 0.9, "probs": {}, "part": "leaf"}

    monkeypatch.setattr(prescribe, "get_diagnosis", _diag)
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [])
    monkeypatch.setattr(prescribe, "get_forecast",
                        lambda: {"next_temp": 30.0, "trend": "유지", "humidity_risk": "높음",
                                "humidity_mean": 90.0, "recent_temp": 29.0})
    with patch("ollama.chat", return_value={"message": {"role": "assistant", "content": _FINAL}}) as mock_chat:
        p = prescribe.prescribe_fast("이 잎 봐줘", image_path="x.jpg")
    assert calls["n"] == 1
    assert isinstance(p, Prescription)
    # ollama.chat은 최종 구조화 JSON 1회만 호출됨(format 포함)
    assert mock_chat.call_count == 1
    assert mock_chat.call_args.kwargs["format"] == Prescription.model_json_schema()


def test_fast_uses_writer_model_env(monkeypatch):
    """OLLAMA_WRITER_MODEL 설정 시 그 모델명으로 최종 chat 호출."""
    monkeypatch.setattr(prescribe, "WRITER_MODEL", "exaone3.5:2.4b")
    with patch("ollama.chat", return_value={"message": {"role": "assistant", "content": _FINAL}}) as mock_chat:
        prescribe.prescribe_fast("오이 병도 알려줘")  # 이미지 없음 → 진단 생략
    assert mock_chat.call_args.kwargs["model"] == "exaone3.5:2.4b"


def test_fast_falls_back_to_model_when_writer_unset(monkeypatch):
    """OLLAMA_WRITER_MODEL 미설정 시 MODEL로 폴백."""
    monkeypatch.setattr(prescribe, "WRITER_MODEL", prescribe.MODEL)
    with patch("ollama.chat", return_value={"message": {"role": "assistant", "content": _FINAL}}) as mock_chat:
        prescribe.prescribe_fast("오이 병도 알려줘")
    assert mock_chat.call_args.kwargs["model"] == prescribe.MODEL


def test_fast_on_progress_order(monkeypatch):
    """on_progress가 diagnosis→rag→forecast→writing 순서로 호출된다."""
    monkeypatch.setattr(prescribe, "get_diagnosis",
                        lambda image_path: {"ood_blocked": False, "label": "leaf_mold",
                                            "prob": 0.9, "probs": {}, "part": "leaf"})
    monkeypatch.setattr(prescribe, "retrieve", lambda q, disease=None, k=3: [
        {"title": "잎곰팡이병 방제", "source": "https://ncpms.rda.go.kr/", "text": "환기하라"}])
    monkeypatch.setattr(prescribe, "get_forecast",
                        lambda: {"next_temp": 30.0, "trend": "유지", "humidity_risk": "높음",
                                "humidity_mean": 90.0, "recent_temp": 29.0})
    stages: list[str] = []
    with patch("ollama.chat", return_value={"message": {"role": "assistant", "content": _FINAL}}):
        prescribe.prescribe_fast("이 잎 봐줘", image_path="x.jpg", on_progress=stages.append)
    assert stages == ["diagnosis", "rag", "forecast", "writing"]


def test_fast_ood_blocked_skips_rag_and_guards(monkeypatch):
    """진단 차단 시 RAG를 호출하지 않고 재촬영을 안내하는 guard가 주입된다."""
    monkeypatch.setattr(prescribe, "get_diagnosis",
                        lambda image_path: {"ood_blocked": True, "reason": "잎이 아닌 부위"})
    called = {"n": 0}

    def _rt(q, disease=None, k=3):
        called["n"] += 1
        return []

    monkeypatch.setattr(prescribe, "retrieve", _rt)
    with patch("ollama.chat", return_value={"message": {"role": "assistant", "content": _FINAL}}) as mock_chat:
        prescribe.prescribe_fast("봐줘", image_path="x.jpg")
    assert called["n"] == 0
    messages = mock_chat.call_args.kwargs["messages"]
    assert any("재촬영" in m.get("content", "") for m in messages)


def test_fast_diagnosis_failure_guards_no_assertion(monkeypatch):
    """get_diagnosis 호출이 예외를 던지면 error로 잡아 병명 단정을 막는 guard가 들어간다."""
    def _boom(image_path):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(prescribe, "get_diagnosis", _boom)
    with patch("ollama.chat", return_value={"message": {"role": "assistant", "content": _FINAL}}) as mock_chat:
        p = prescribe.prescribe_fast("이 잎 봐줘", image_path="x.jpg")
    assert isinstance(p, Prescription)
    messages = mock_chat.call_args.kwargs["messages"]
    assert any("재시도" in m.get("content", "") for m in messages)


def test_fast_schema_violation_retries_then_falls_back():
    """스키마 위반 응답이 2회 연속이면 안전 폴백 Prescription을 반환한다."""
    bad = {"message": {"role": "assistant", "content": "not json"}}
    with patch("ollama.chat", side_effect=[bad, bad]) as mock_chat:
        p = prescribe.prescribe_fast("오이 병도 알려줘")
    assert p.진단요약.startswith("처방 생성에 실패")
    assert mock_chat.call_count == 2


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
