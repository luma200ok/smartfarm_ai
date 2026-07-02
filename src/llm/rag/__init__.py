"""
RAG 검색 진입점 — 진단 라벨로 스코프한 농사로 재배가이드 chunk 검색.

retrieve(query, disease, k) → [{text, title, source, source_name, disease, score}]
prescribe.py 가 이 결과를 처방 근거(system 메시지)로 주입하고 근거출처를 채운다.

백엔드는 RAG_BACKEND env로 분기(memory 기본 | pgvector). pgvector 실패 시 memory로 폴백
— DB 장애가 처방 흐름을 막지 않도록(notify.py와 동일한 best-effort 원칙).
"""
import logging
import os

from . import pg_store
from .corpus import load_chunks
from .store import build_index, search

_log = logging.getLogger(__name__)


def _backend() -> str:
    # 호출 시점(import 시점 아님)에 읽는다 — prescribe.py가 load_dotenv 전에 이 모듈을
    # import하므로 모듈 상수로 캐싱하면 서버 .env의 RAG_BACKEND가 무시됨.
    return os.getenv("RAG_BACKEND", "memory")


def retrieve(query: str, disease: str | None = None, k: int = 3) -> list[dict]:
    if _backend() == "pgvector":
        try:
            return pg_store.search(query, disease=disease, k=k)
        except Exception as e:                       # DB 장애 → memory 폴백(예외 전파 금지)
            _log.warning("pgvector 검색 실패 — memory 폴백: %s", e)

    chunks = load_chunks()
    if not chunks:
        return []
    emb = build_index(chunks)
    return search(query, chunks, emb, disease=disease, k=k)
