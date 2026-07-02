"""
PostgreSQL 공용 연결 헬퍼 — RAG(pgvector) 저장소·처방/경보 이력이 공유.

notify.py 와 동일 계약: DATABASE_URL 미설정 → 조용히 비활성(None), 예외를 밖으로 전파하지 않음.
과설계 배제: 커넥션풀 없이 connect-per-call(로컬 소켓 few ms) — 청크 수십 개 규모라 불필요.
"""
import logging
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

# override=True: notify.py 와 동일 — import 순서 무관하게 프로젝트 .env 가 우선
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

_log = logging.getLogger(__name__)
_CONNECT_TIMEOUT = 3   # PG 다운 시 hang 방지(초)


def get_conn() -> psycopg.Connection | None:
    """DATABASE_URL 미설정 시 None(조용한 비활성). 연결 실패 시 예외 전파 — 호출자가 처리."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    conn = psycopg.connect(url, autocommit=True, connect_timeout=_CONNECT_TIMEOUT)
    register_vector(conn)
    return conn
