"""src/llm/rag/pg_store.py — pgvector 검색 백엔드(단위=mock) + integration(실 PG).

단위 테스트는 db.get_conn()을 몽키패치해 cursor/fetchall을 mock — PG 서버 불필요.
integration은 DATABASE_URL 설정 시에만 schema.sql 적용 → 실 INSERT/검색.
"""
import os

import numpy as np
import pytest

from llm.rag import pg_store


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def test_search_returns_none_conn_as_empty(monkeypatch):
    """DATABASE_URL 미설정(get_conn()→None) → 조용히 []."""
    monkeypatch.setattr(pg_store.db, "get_conn", lambda: None)
    assert pg_store.search("아무거나") == []


def test_search_maps_rows_to_dict(monkeypatch):
    rows = [("환기하라", "가이드", "https://x", "농사로", "leaf_mold", 0.92)]
    monkeypatch.setattr(pg_store.db, "get_conn", lambda: _FakeConn(rows))
    monkeypatch.setattr(pg_store.store, "_embed", lambda texts: np.array([[1.0, 0.0]], dtype="float32"))
    out = pg_store.search("곰팡이", disease="leaf_mold", k=3)
    assert out == [{"text": "환기하라", "title": "가이드", "source": "https://x",
                    "source_name": "농사로", "disease": "leaf_mold", "score": 0.92}]


def test_search_disease_filter_included_in_query(monkeypatch):
    fake = _FakeConn([])
    monkeypatch.setattr(pg_store.db, "get_conn", lambda: fake)
    monkeypatch.setattr(pg_store.store, "_embed", lambda texts: np.array([[1.0, 0.0]], dtype="float32"))
    out = pg_store.search("q", disease="tylcv", k=2)
    assert out == []          # 매칭 없으면 빈 리스트(엉뚱한 병해 근거 방지)


def test_search_no_disease_filter(monkeypatch):
    rows = [("text1", "t", "", "", "", 0.5)]
    monkeypatch.setattr(pg_store.db, "get_conn", lambda: _FakeConn(rows))
    monkeypatch.setattr(pg_store.store, "_embed", lambda texts: np.array([[1.0, 0.0]], dtype="float32"))
    out = pg_store.search("q", disease=None, k=1)
    assert len(out) == 1 and out[0]["text"] == "text1"


# ── retrieve() 백엔드 분기 (rag/__init__.py) ────────────────────────────
def test_retrieve_defaults_to_memory(monkeypatch):
    from llm import rag
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    called = {"pg": False}
    monkeypatch.setattr(rag.pg_store, "search", lambda *a, **k: called.__setitem__("pg", True) or [])
    monkeypatch.setattr(rag, "load_chunks", lambda: [])
    rag.retrieve("q")
    assert called["pg"] is False


def test_retrieve_uses_pgvector_when_backend_env_set(monkeypatch):
    from llm import rag
    monkeypatch.setenv("RAG_BACKEND", "pgvector")
    monkeypatch.setattr(rag.pg_store, "search",
                        lambda query, disease=None, k=3: [{"text": "pg결과"}])
    out = rag.retrieve("q")
    assert out == [{"text": "pg결과"}]


def test_retrieve_falls_back_to_memory_on_pg_error(monkeypatch):
    from llm import rag
    monkeypatch.setenv("RAG_BACKEND", "pgvector")

    def _raise(*a, **k):
        raise RuntimeError("PG 다운")

    monkeypatch.setattr(rag.pg_store, "search", _raise)
    monkeypatch.setattr(rag, "load_chunks", lambda: [])
    out = rag.retrieve("q")
    assert out == []          # memory 경로로 폴백(코퍼스 없음 → 빈 리스트) — 예외 전파 안 됨


# ══════════════════════════════════════════════════════════════════════
# integration — 실 PostgreSQL 필요. DATABASE_URL 미설정 시 skip.
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def pg_conn():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL 미설정 — integration 스킵")
    from llm import db as db_mod
    ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
    conn = db_mod.get_conn()
    with open(ROOT / "db" / "schema.sql", encoding="utf-8") as f:
        conn.execute(f.read())
    conn.execute("TRUNCATE rag_chunks, rag_meta")
    yield conn
    conn.close()


@pytest.mark.integration
def test_integration_search_disease_scope(monkeypatch, pg_conn):
    from llm.rag import pg_store as pgs
    emb = [0.1] * 1024
    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rag_chunks (text, title, source, source_name, disease, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            ("환기하라", "가이드", "", "", "leaf_mold", emb),
        )
    monkeypatch.setattr(pgs.store, "_embed", lambda texts: np.array([emb], dtype="float32"))
    out = pgs.search("q", disease="leaf_mold", k=3)
    assert len(out) == 1 and out[0]["disease"] == "leaf_mold"

    out_empty = pgs.search("q", disease="tylcv", k=3)
    assert out_empty == []


@pytest.mark.integration
def test_integration_sync_loads_corpus_and_is_idempotent(monkeypatch, pg_conn):
    """sync.py — 코퍼스 적재 + 재실행 시 스킵(코퍼스 불변) + --force 강제 재적재."""
    from llm.rag import sync as sync_mod

    fake_chunks = [
        {"text": "환기하라", "title": "가이드", "source": "", "source_name": "", "disease": "leaf_mold"},
        {"text": "물을 준다", "title": "가이드2", "source": "", "source_name": "", "disease": "normal"},
    ]
    monkeypatch.setattr(sync_mod, "load_chunks", lambda: fake_chunks)
    monkeypatch.setattr(sync_mod, "_embed", lambda texts: np.ones((len(texts), 1024), dtype="float32"))

    n1 = sync_mod.sync()
    assert n1 == 2
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_chunks")
        assert cur.fetchone()[0] == 2

    n2 = sync_mod.sync()          # 코퍼스 불변 → 스킵
    assert n2 == 0

    n3 = sync_mod.sync(force=True)  # 강제 재적재
    assert n3 == 2
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_chunks")
        assert cur.fetchone()[0] == 2   # DELETE 후 재삽입 — 중복 아님


@pytest.mark.integration
def test_integration_sync_no_database_url_raises(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from llm.rag import sync as sync_mod
    with pytest.raises(RuntimeError):
        sync_mod.sync()
