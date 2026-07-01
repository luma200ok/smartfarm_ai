"""src/llm/rag/store.py — 코사인 검색·disease 스코프·캐시 (임베딩 모킹)."""
import numpy as np

from llm.rag import store

_CHUNKS = [
    {"text": "잎곰팡이 곰팡이", "disease": "leaf_mold", "title": "A", "source": ""},
    {"text": "황화 바이러스", "disease": "tylcv", "title": "B", "source": ""},
    {"text": "물주기 관리", "disease": "normal", "title": "C", "source": ""},
]
_EMB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")


def test_search_ranks_top_by_cosine(monkeypatch):
    # 쿼리 임베딩을 leaf_mold 축([1,0,0])으로 고정 → A 가 1위
    monkeypatch.setattr(store, "_embed", lambda t: np.array([[1, 0, 0]], dtype="float32"))
    res = store.search("곰팡이?", _CHUNKS, _EMB, disease=None, k=1)
    assert res[0]["title"] == "A"


def test_search_scopes_by_disease(monkeypatch):
    monkeypatch.setattr(store, "_embed", lambda t: np.array([[1, 0, 0]], dtype="float32"))
    res = store.search("아무거나", _CHUNKS, _EMB, disease="tylcv", k=3)
    assert res and all(r["disease"] == "tylcv" for r in res)


def test_search_unknown_disease_returns_empty(monkeypatch):
    monkeypatch.setattr(store, "_embed", lambda t: np.array([[1, 0, 0]], dtype="float32"))
    assert store.search("x", _CHUNKS, _EMB, disease="ghost") == []


def test_build_index_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "INDEX_PATH", tmp_path / "idx.npz")
    calls = {"n": 0}

    def _emb(texts):
        calls["n"] += 1
        return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(store, "_embed", _emb)
    chunks = [{"text": "a"}, {"text": "b"}]
    store.build_index(chunks)
    store.build_index(chunks)             # 같은 코퍼스 → 캐시 사용, 재임베딩 없음
    assert calls["n"] == 1


def test_build_index_reembeds_on_corpus_change(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "INDEX_PATH", tmp_path / "idx.npz")
    calls = {"n": 0}

    def _emb(texts):
        calls["n"] += 1
        return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr(store, "_embed", _emb)
    store.build_index([{"text": "a"}, {"text": "b"}])
    store.build_index([{"text": "a"}, {"text": "b"}, {"text": "c"}])   # 문서 추가 → 재임베딩
    assert calls["n"] == 2
