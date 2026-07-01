"""
임베딩 + 코사인 검색 — Ollama bge-m3 로 chunk 를 벡터화하고 numpy 로 top-k 검색.

벡터 DB 없이(의존성 0) 동작. 임베딩은 코퍼스 내용 해시로 디스크 캐시(.rag_index.npz):
코퍼스가 바뀌면 재임베딩, 아니면 즉시 로드.
"""
import hashlib
import os

import numpy as np
import ollama

from .corpus import CORPUS_DIR

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
INDEX_PATH = CORPUS_DIR / ".rag_index.npz"


def _embed(texts: list[str]) -> np.ndarray:
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.asarray(resp["embeddings"], dtype="float32")


def _corpus_key(chunks: list[dict]) -> str:
    h = hashlib.sha256(EMBED_MODEL.encode())
    h.update(len(chunks).to_bytes(8, "big"))         # chunk 개수 반영
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x1e")                            # chunk 경계 구분자(연결 충돌 방지)
    return h.hexdigest()


def build_index(chunks: list[dict]) -> np.ndarray:
    """chunk 임베딩 행렬(N×D). 캐시 해시+행수가 맞으면 디스크에서 로드, 아니면 재임베딩."""
    key = _corpus_key(chunks)
    if INDEX_PATH.exists():
        cached = np.load(INDEX_PATH, allow_pickle=False)
        # shape 검증 — 캐시가 손상/불일치면 재빌드(엉뚱한 chunk-임베딩 오매칭 방지)
        if str(cached["key"].item()) == key and cached["emb"].shape[0] == len(chunks):
            return cached["emb"]
    emb = _embed([c["text"] for c in chunks])
    np.savez(INDEX_PATH, emb=emb, key=np.array(key))
    return emb


def search(query: str, chunks: list[dict], emb: np.ndarray,
           disease: str | None = None, k: int = 3) -> list[dict]:
    """disease 로 1차 필터 후 코사인 유사도 top-k. 매칭 chunk 없으면 [] (엉뚱한 병해 근거 방지)."""
    idxs = [i for i, c in enumerate(chunks)
            if disease is None or c.get("disease") == disease]
    if not idxs:
        return []
    q = _embed([query])[0]
    sub = emb[idxs]
    scores = (sub @ q) / (np.linalg.norm(sub, axis=1) * np.linalg.norm(q) + 1e-8)
    order = np.argsort(-scores)[:k]
    return [{**chunks[idxs[o]], "score": float(scores[o])} for o in order]
