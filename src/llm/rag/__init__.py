"""
RAG 검색 진입점 — 진단 라벨로 스코프한 농사로 재배가이드 chunk 검색.

retrieve(query, disease, k) → [{text, title, source, source_name, disease, score}]
prescribe.py 가 이 결과를 처방 근거(system 메시지)로 주입하고 근거출처를 채운다.
"""
from .corpus import load_chunks
from .store import build_index, search


def retrieve(query: str, disease: str | None = None, k: int = 3) -> list[dict]:
    chunks = load_chunks()
    if not chunks:
        return []
    emb = build_index(chunks)
    return search(query, chunks, emb, disease=disease, k=k)
