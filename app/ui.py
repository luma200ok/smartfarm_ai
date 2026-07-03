"""
공통 UI 헬퍼 — 이슈 #10 Streamlit 앱 전면 정리.

모든 app/views/*.py 페이지가 공유하는 헤더·경보·상태뱃지·불가안내 렌더 유틸.
디자인 원칙: 이모지는 타이틀·섹션 헤더에만 1개, 본문 내부는 제거. 사용자 대면 문구는 "~해요"체.
"""
import streamlit as st

_ALERT_BOX = {"경고": st.error, "주의": st.warning, "정보": st.info}


def inject_css():
    """초록 테마 공통 CSS — 전 페이지 승격(구 phase1_ml.py 탭 스타일)."""
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 12px 24px;
        background-color: #F1F8E9;
        color: #2E5A1C;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4C9A2A;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, caption: str | None = None):
    """페이지 최상단 타이틀 + 설명 캡션(각 view의 render() 첫 줄에서 호출)."""
    st.title(title)
    if caption:
        st.caption(caption)


def section(title: str, caption: str | None = None):
    """섹션 소제목 + 부연 캡션."""
    st.subheader(title)
    if caption:
        st.caption(caption)


def metric_row(items: list[tuple[str, str, str | None]]):
    """내부 vs 외기 대비 등 metric 카드를 한 줄로. items=[(label, value, delta), ...]."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        if delta:
            col.metric(label, value, delta)
        else:
            col.metric(label, value)


def status_badge(ok: bool, label: str):
    """단일 상태 뱃지 — 정상(success)/오프라인(warning) 표시."""
    (st.success if ok else st.warning)(label)


def unavailable(feature: str, reason: str, hint: str | None = None):
    """모델·데이터 없음 등 선택 기능 비활성 안내 — 단일 포맷으로 통일."""
    msg = f"ℹ️ {feature} 사용 불가 — {reason}"
    if hint:
        msg += f" ({hint})"
    st.caption(msg)


def alert_box(level: str, text: str):
    """경고/주의/정보 3단계 알림 — 구 phase3의 중복 dict 매핑 흡수."""
    box = _ALERT_BOX.get(level, st.info)
    box(text)
