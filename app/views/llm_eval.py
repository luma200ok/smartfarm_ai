"""
LLM 실험 기록 페이지 — [프로젝트 기록] 그룹. 재배 도우미(처방·코치·RAG 상담) 실험 기록(Phase 3).

ML/DL과 달리 혼동행렬·ROC 같은 정량 figure가 없는 도메인(로컬 LLM 처방·RAG 상담)이라
서술형 결과·처방 지연 개선 수치·RAG 코퍼스 통계로 구성한다(이슈 #52). 서술·수치 출처는
docs/phase3_llm.md §9(결과)·§10(배운점)·docs/STATUS.md — 문서에 없는 값은 지어내지 않는다.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ui import kpi_cards, page_header, unavailable

FIG_EXPECT = ROOT / "docs" / "figures" / "expect_regression"


def _corpus_stats():
    """농사로 코퍼스 문서·chunk 통계 — 임베딩 호출 없이(Ollama 불필요) 로컬 md만 파싱해서
    죽지 않는다. corpus.py 임포트 자체가 실패해도(의존성 변경 등) None으로 조용히 폴백."""
    try:
        from llm.rag.corpus import CORPUS_DIR, load_chunks
    except Exception:
        return None

    if not CORPUS_DIR.is_dir():
        return None
    chunks = load_chunks(CORPUS_DIR)
    if not chunks:
        return None
    diseases = sorted({c.get("disease", "") for c in chunks if c.get("disease")})
    return {
        "docs": len(list(CORPUS_DIR.glob("*.md"))),
        "chunks": len(chunks),
        "diseases": diseases,
    }


def _embed_dim():
    """캐시된 임베딩 인덱스(data/nongsaro/.rag_index.npz)가 있으면 차원 수만 읽는다.
    임베딩 자체는 Ollama(bge-m3) 호출이 필요해 이 페이지에서 새로 만들지 않는다."""
    try:
        from llm.rag.store import INDEX_PATH
    except Exception:
        return None
    if not INDEX_PATH.exists():
        return None
    try:
        import numpy as np
        data = np.load(INDEX_PATH, allow_pickle=False)
        return int(data["emb"].shape[1])
    except Exception:
        return None


def render():
    page_header(
        "📊 LLM 실험 기록",
        "Ollama 로컬 LLM(qwen2.5:14b) 기반 재배 도우미(처방·코치·RAG 상담) 실험 기록(Phase 3) — "
        "정량 figure 대신 서술형 결과·지연 개선 수치·RAG 코퍼스 통계로 구성해요.",
    )

    st.subheader("재배 도우미 결과 요약")
    st.markdown(
        "- **처방(C)**: 잎 사진 → 게이트→진단→function calling→RAG 근거→구조화 JSON 처방. "
        "근거출처는 LLM이 아니라 **코드가 채워** 환각을 배제하고, 스키마 위반 시 1회 재시도 후 "
        "안전 폴백해요.\n"
        "- **일일 코치(A)·조기 경보(B)**: LSTM 환경 예측을 기반으로, 습도 위험이 있을 때 RAG 근거를 "
        "경보에 주입해요.\n"
        "- **시간축 처방**: 습도 민감 병해(잎곰팡이병·잎마름역병) 진단 시 환경 예측을 교차 주입 — "
        "ML+DL+LLM이 한 처방에서 만나요.\n"
        "- **환각 방어 3종** — ① 확신도(prob) 구간별 톤 분기(고확신 단정 / 애매 확인 권유) "
        "② 게이트 차단 시 원인·재촬영법을 함께 안내 ③ 모델이 아는 클래스 범위(잎병 3종+정상)를 "
        "system 프롬프트에 고정해 범위 밖 질문은 정직하게 '진단 보류'로 답해요.\n"
        "- **자동 감시**: 가상센서 스트림을 재생해 임계 판정 후 디스코드 Webhook으로 알림(레벨 전이 "
        "시만 발송, 전송 실패 시 재시도)해요."
    )

    st.subheader("처방 지연 개선")
    kpi_cards([
        {"label": "로컬 웜 처방(이슈 #15)", "value": "17~19", "unit": "초",
         "chip": "28초 → 약 -35%", "chip_level": "ok", "icon": "⏱️"},
        {"label": "서버 fast-path(이슈 #18)", "value": "16.2", "unit": "초",
         "chip": "342.6초 → -95%", "chip_level": "ok", "icon": "🚀"},
    ])
    st.caption(
        "로컬(웜 기준): tool 라운드 num_predict 캡 + 모든 Ollama 호출 keep_alive=30m(콜드 재적재 "
        "제거) + 프롬프트 다이어트(-17~20% 토큰). "
        "서버(OCI 3코어 CPU): 진단·RAG·예보를 코드가 직접 실행하고 LLM은 최종 JSON 처방만 1-call로 "
        "작성하는 fast-path + writer 모델 분리(서버 exaone3.5:2.4b)."
    )

    st.subheader("RAG 코퍼스 통계")
    stats = _corpus_stats()
    if stats:
        items = [
            {"label": "코퍼스 문서", "value": str(stats["docs"]), "unit": "개", "icon": "📖"},
            {"label": "검색 chunk", "value": str(stats["chunks"]), "unit": "개", "icon": "🧩"},
            {"label": "병해 커버리지", "value": str(len(stats["diseases"])), "unit": "종", "icon": "🌿"},
        ]
        dim = _embed_dim()
        if dim:
            items.append({"label": "임베딩 차원", "value": str(dim), "unit": "d", "icon": "🔢"})
        kpi_cards(items)
        if stats["diseases"]:
            st.caption(f"disease 태그: {', '.join(stats['diseases'])} · 임베딩 모델 bge-m3(Ollama)")
        if not dim:
            unavailable("임베딩 차원 표시", "캐시 인덱스(.rag_index.npz) 없음",
                        "AI 처방 페이지 최초 호출 시 생성")
    else:
        unavailable("RAG 코퍼스 통계", "data/nongsaro/*.md 없음")

    if FIG_EXPECT.is_dir() and any(FIG_EXPECT.glob("*.png")):
        st.subheader("통합 기대값 회귀 (외기 → 실내, 조기 경보 입력)")
        st.caption(
            "LLM 조기 경보(B)가 참조하는 외기→실내 기대값 회귀 — XGBoost, GroupKFold MAE 1.11/1.44℃."
        )
        captions = [
            ("model_compare.png", "RF vs XGB GroupKFold MAE 비교"),
            ("residual_hist.png", "잔차 분포"),
            ("feature_importance.png", "XGB 피처 중요도"),
        ]
        cols = st.columns(len(captions))
        for col, (name, cap) in zip(cols, captions):
            p = FIG_EXPECT / name
            if p.exists():
                col.image(str(p), caption=cap, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="LLM 실험 기록", page_icon="📊", layout="wide")
    render()
