"""src/llm/history.py — 처방/경보 이력 저장(best-effort). no-op·예외삼킴 검증(단위) + 실 INSERT(integration)."""
import os

import pytest
from pydantic import BaseModel

from llm import history


class _FakePresc(BaseModel):
    진단요약: str = "잎곰팡이병 의심"


class _FakeCursor:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.store.append((sql, params))


class _FakeConn:
    def __init__(self):
        self.inserted = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):          # psycopg Connection 컨텍스트매니저 — 종료 시 close
        self.closed = True
        return False

    def cursor(self):
        return _FakeCursor(self.inserted)


# ── no-op (DATABASE_URL 미설정 = get_conn() → None) ─────────────────
def test_save_prescription_noop_when_no_conn(monkeypatch):
    monkeypatch.setattr(history.db, "get_conn", lambda: None)
    ok = history.save_prescription("메시지", None, None, _FakePresc())
    assert ok is False


def test_save_alert_noop_when_no_conn(monkeypatch):
    monkeypatch.setattr(history.db, "get_conn", lambda: None)
    ok = history.save_alert("monitor", "경고", "leaf_mold", "고습", {})
    assert ok is False


def test_save_chat_noop_when_no_conn(monkeypatch):
    monkeypatch.setattr(history.db, "get_conn", lambda: None)
    ok = history.save_chat("질문", "답변", [])
    assert ok is False


# ── 예외 삼킴 (DB 장애가 처방/경보 흐름을 막지 않음) ──────────────────
def test_save_prescription_swallows_exception(monkeypatch):
    def _raise():
        raise RuntimeError("PG 다운")

    monkeypatch.setattr(history.db, "get_conn", _raise)
    ok = history.save_prescription("메시지", None, None, _FakePresc())
    assert ok is False


def test_save_alert_swallows_exception(monkeypatch):
    def _raise():
        raise RuntimeError("PG 다운")

    monkeypatch.setattr(history.db, "get_conn", _raise)
    ok = history.save_alert("early_warning", "주의", "leaf_mold", "고습", {})
    assert ok is False


def test_save_chat_swallows_exception(monkeypatch):
    def _raise():
        raise RuntimeError("PG 다운")

    monkeypatch.setattr(history.db, "get_conn", _raise)
    ok = history.save_chat("질문", "답변", [])
    assert ok is False


# ── 정상 경로(mock conn) — INSERT 1회 호출 확인 ───────────────────────
def test_save_prescription_inserts_when_conn_available(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    ok = history.save_prescription("메시지", "x.jpg", {"label": "leaf_mold"}, _FakePresc())
    assert ok is True
    assert len(conn.inserted) == 1
    assert "INSERT INTO prescriptions" in conn.inserted[0][0]


def test_save_prescription_inserts_caller_ref_when_given(monkeypatch):
    """smartfarm_ai#66 — caller_ref가 INSERT 파라미터 끝에 그대로 실린다."""
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    ok = history.save_prescription("메시지", None, None, _FakePresc(), caller_ref="tenant-1")
    assert ok is True
    assert conn.inserted[0][1][-1] == "tenant-1"


def test_save_prescription_caller_ref_default_none(monkeypatch):
    """caller_ref 미전달 시 기존과 동일하게 None으로 저장된다."""
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    history.save_prescription("메시지", None, None, _FakePresc())
    assert conn.inserted[0][1][-1] is None


def test_save_alert_inserts_when_conn_available(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    ok = history.save_alert("monitor", "경고", "leaf_mold", "고습", {"key": "humidity"})
    assert ok is True
    assert len(conn.inserted) == 1
    assert "INSERT INTO alerts" in conn.inserted[0][0]


# ── smartfarm_ai#84: 챗(RAG 자유 질의) 이력 ─────────────────────────────
def test_save_chat_inserts_when_conn_available(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    ok = history.save_chat("질문", "답변", ["출처1"])
    assert ok is True
    assert len(conn.inserted) == 1
    assert "INSERT INTO chat_messages" in conn.inserted[0][0]


def test_save_chat_inserts_caller_ref_when_given(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    ok = history.save_chat("질문", "답변", [], caller_ref="tenant-1")
    assert ok is True
    assert conn.inserted[0][1][-1] == "tenant-1"


def test_save_chat_caller_ref_default_none(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn)
    history.save_chat("질문", "답변", [])
    assert conn.inserted[0][1][-1] is None


def test_save_closes_connection(monkeypatch):
    """connect-per-call — 저장 후 커넥션이 닫혀야 한다(monitor tight loop 누수 방지, P2 픽스)."""
    conn1, conn2 = _FakeConn(), _FakeConn()
    monkeypatch.setattr(history.db, "get_conn", lambda: conn1)
    history.save_prescription("m", None, None, _FakePresc())
    assert conn1.closed is True

    monkeypatch.setattr(history.db, "get_conn", lambda: conn2)
    history.save_alert("monitor", "경고", "", "r", {})
    assert conn2.closed is True


# ── prescribe/pipeline/monitor 훅이 기존 흐름을 깨지 않는지 (history mock 호출 assert) ──
def test_prescribe_hook_calls_history(monkeypatch):
    import json
    from unittest.mock import patch

    from llm import prescribe

    final = json.dumps({
        "진단요약": "정상", "원인": "-", "즉시조치": "-", "예방": "-", "재촬영시점": "-", "근거출처": [],
    }, ensure_ascii=False)
    responses = [
        {"message": {"role": "assistant", "content": "", "tool_calls": []}},
        {"message": {"role": "assistant", "content": final}},
    ]
    calls = {}
    monkeypatch.setattr(prescribe.history, "save_prescription",
                        lambda *a, **k: calls.setdefault("called", True))
    with patch("ollama.chat", side_effect=responses):
        prescribe.prescribe("범위 밖 질문")
    assert calls.get("called") is True


def test_pipeline_early_warning_hook_calls_history(monkeypatch):
    import json
    from unittest.mock import patch

    from llm import pipeline

    monkeypatch.setattr(pipeline, "get_forecast",
                        lambda window=None: {"next_temp": 30.0, "trend": "유지",
                                             "humidity_risk": "높음", "humidity_mean": 92.0})
    monkeypatch.setattr(pipeline, "retrieve", lambda q, disease=None, k=3: [])
    calls = {}
    monkeypatch.setattr(pipeline.history, "save_alert",
                        lambda *a, **k: calls.setdefault("called", True))
    out = json.dumps({"경보수준": "주의", "위험병해": "잎곰팡이병", "이유": "고습", "권장조치": "환기"},
                     ensure_ascii=False)
    with patch("ollama.chat", return_value={"message": {"content": out}}):
        pipeline.early_warning()
    assert calls.get("called") is True


def test_monitor_run_hook_calls_history(monkeypatch):
    from llm import monitor

    class FakeVS:
        def __init__(self, year=None):
            self.series = list(range(1 + monitor.infer.WINDOW - 1))
            self.farm = ("테스트",)
            self.i = 0

        def reading(self):
            return {"습도내부_평균": 95.0, "온도내부_평균": 25.0}

        def date(self):
            return "d0"

        def window(self):
            return None

        def tick(self):
            self.i += 1

    monkeypatch.setattr(monitor, "VirtualSensor", FakeVS)
    monkeypatch.setattr(monitor.notify, "send_discord", lambda e: (True, "ok"))
    calls = {}
    monkeypatch.setattr(monitor.history, "save_alert",
                        lambda *a, **k: calls.setdefault("called", True))
    monitor.run(interval=0)
    assert calls.get("called") is True


# ══════════════════════════════════════════════════════════════════════
# integration — 실 PostgreSQL 필요. DATABASE_URL 미설정 시 skip.
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def pg_conn():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL 미설정 — integration 스킵")
    from pathlib import Path

    from llm import db as db_mod
    ROOT = Path(__file__).resolve().parents[1]
    conn = db_mod.get_conn()
    with open(ROOT / "db" / "schema.sql", encoding="utf-8") as f:
        conn.execute(f.read())
    conn.execute("TRUNCATE prescriptions, alerts, chat_messages")
    yield conn
    conn.close()


@pytest.mark.integration
def test_integration_save_prescription_and_alert(pg_conn):
    ok1 = history.save_prescription("실제 메시지", None, {"label": "leaf_mold"}, _FakePresc())
    assert ok1 is True
    ok2 = history.save_alert("monitor", "경고", "leaf_mold", "고습", {"key": "humidity"})
    assert ok2 is True

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM prescriptions")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM alerts")
        assert cur.fetchone()[0] == 1


# ── smartfarm_ai#84 후속(python-reviewer P3): save_chat 실 Postgres 라운드트립 ──────────
# sources(list[str]) ↔ JSONB 직렬화(Jsonb 저장)·역직렬화(psycopg3 jsonb 로더 → 다시 list)를
# 목으로는 못 잡는다 — 실제로 다시 읽어 타입·값·순서까지 보존되는지 검증한다(레포 룰:
# 직렬화 변경은 실 DB 라운드트립 검증 의무).
@pytest.mark.integration
def test_integration_save_chat_roundtrips_sources_list(pg_conn):
    sources = ["잎곰팡이병 방제 (농촌진흥청) — https://www.nongsaro.go.kr/", "두 번째 출처"]
    ok = history.save_chat("실제 질문", "실제 답변", sources, caller_ref="tenant-1")
    assert ok is True

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chat_messages")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT question, answer, sources, caller_ref FROM chat_messages")
        question, answer, saved_sources, caller_ref = cur.fetchone()
        assert question == "실제 질문"
        assert answer == "실제 답변"
        assert isinstance(saved_sources, list)   # JSONB → psycopg3 로더가 파이썬 list로 역직렬화
        assert saved_sources == sources          # 원소·순서 보존
        assert caller_ref == "tenant-1"


@pytest.mark.integration
def test_integration_save_chat_empty_sources_roundtrips_as_empty_list(pg_conn):
    ok = history.save_chat("질문 없음 출처", "답변", [])
    assert ok is True

    with pg_conn.cursor() as cur:
        cur.execute("SELECT sources, caller_ref FROM chat_messages WHERE question = %s",
                    ("질문 없음 출처",))
        saved_sources, caller_ref = cur.fetchone()
        assert saved_sources == []
        assert caller_ref is None
