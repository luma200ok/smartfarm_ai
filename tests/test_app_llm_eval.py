"""app/views/llm_eval.py — LLM 실험 기록 페이지(이슈 #52) 스모크 + 폴백 경로 테스트.

ML/DL과 달리 정량 figure가 없어 서술형 결과 + RAG 코퍼스 통계로 구성한다. AppTest로 페이지
전체를 렌더해 예외 없이 완주하는지 확인하고(monitor.py 패턴, 이슈 #23), 코퍼스 통계 헬퍼는
data/nongsaro/*.md 실 데이터로 개수를 검증한다. 파일이 없는 환경(CI 등)에서도 unavailable로
조용히 폴백하는지는 monkeypatch로 별도 확인한다."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_EVAL_PAGE = ROOT / "app" / "views" / "llm_eval.py"


def _import_llm_eval_module():
    for p in (ROOT / "src", ROOT / "app", ROOT / "app" / "views"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("llm_eval")


def test_llm_eval_page_renders_without_exception():
    """AppTest로 /llm-eval 스크립트 전체 실행 — RAG 코퍼스 유무와 무관하게 절대 죽지 않아야 한다."""
    for p in (ROOT / "src", ROOT / "app", ROOT / "app" / "views"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(LLM_EVAL_PAGE))
    at.run(timeout=30)

    assert not at.exception, f"페이지 렌더 중 예외 발생: {[str(e) for e in at.exception]}"


def test_corpus_stats_counts_real_nongsaro_docs():
    """data/nongsaro/*.md 실 데이터 기준 문서·chunk·disease 태그 개수 검증(회귀 방지)."""
    llm_eval_mod = _import_llm_eval_module()

    stats = llm_eval_mod._corpus_stats()

    assert stats is not None
    assert stats["docs"] >= 4  # 잎곰팡이병·황화잎말이·잎마름역병·토마토 일반
    assert stats["chunks"] > 0
    assert "tylcv" in stats["diseases"]
    assert "leaf_mold" in stats["diseases"]


def test_corpus_stats_none_when_corpus_dir_missing(monkeypatch, tmp_path):
    """코퍼스 디렉터리가 없는 환경(CI 등)이면 None — 호출측이 unavailable로 폴백한다."""
    llm_eval_mod = _import_llm_eval_module()
    from llm.rag import corpus as corpus_mod

    monkeypatch.setattr(corpus_mod, "CORPUS_DIR", tmp_path / "nongsaro-does-not-exist")

    assert llm_eval_mod._corpus_stats() is None


def test_embed_dim_none_when_index_missing(monkeypatch, tmp_path):
    """캐시 임베딩 인덱스(.rag_index.npz)가 없으면 None — 카드에서 조용히 생략된다."""
    llm_eval_mod = _import_llm_eval_module()
    from llm.rag import store as store_mod

    monkeypatch.setattr(store_mod, "INDEX_PATH", tmp_path / ".rag_index.npz")

    assert llm_eval_mod._embed_dim() is None


def test_embed_dim_reads_cached_index_dimension(monkeypatch, tmp_path):
    """캐시 인덱스가 있으면 임베딩 행렬의 두 번째 축(차원 수)을 반환한다."""
    import numpy as np

    llm_eval_mod = _import_llm_eval_module()
    from llm.rag import store as store_mod

    index_path = tmp_path / ".rag_index.npz"
    np.savez(index_path, emb=np.zeros((15, 1024), dtype="float32"), key=np.array("dummy"))
    monkeypatch.setattr(store_mod, "INDEX_PATH", index_path)

    assert llm_eval_mod._embed_dim() == 1024


def test_render_falls_back_to_unavailable_when_corpus_stats_none(monkeypatch):
    """_corpus_stats()가 None이면 RAG 통계 kpi_cards 대신 unavailable 안내가 호출된다
    (처방 지연 개선 섹션의 kpi_cards 호출은 코퍼스와 무관하므로 그대로 발생, 무크래시 원칙)."""
    llm_eval_mod = _import_llm_eval_module()

    unavailable_calls = []
    kpi_calls = []
    monkeypatch.setattr(llm_eval_mod, "_corpus_stats", lambda: None)
    monkeypatch.setattr(llm_eval_mod, "unavailable",
                         lambda feature, reason, hint=None: unavailable_calls.append(feature))
    monkeypatch.setattr(llm_eval_mod, "kpi_cards", lambda items: kpi_calls.append(items))
    monkeypatch.setattr(llm_eval_mod, "page_header", lambda *a, **k: None)
    monkeypatch.setattr(llm_eval_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(llm_eval_mod.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(llm_eval_mod.st, "caption", lambda *a, **k: None)

    llm_eval_mod.render()

    assert "RAG 코퍼스 통계" in unavailable_calls
    # 처방 지연 개선 kpi_cards(1회)만 호출되고, RAG 코퍼스 카드는 호출되지 않는다.
    assert len(kpi_calls) == 1
    assert all("코퍼스 문서" != item.get("label") for item in kpi_calls[0])
