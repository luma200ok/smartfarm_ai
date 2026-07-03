"""
농장 대시보드 페이지 — [서비스] 그룹 기본(default) 페이지.

상단 경보 배너 + 핵심 지표 카드 4장 + 최근 7일 실측 vs 기대값 미니 차트 + 기능 바로가기 카드.
모델·데이터가 없는 환경에서도 절대 죽지 않도록 모든 데이터 접근을 try/exists 가드로 감싼다(이슈 #10 C4).
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from state import get_vsensor
from ui import metric_row, page_header, section, unavailable


def _latest_vsensor():
    """가장 최근 연도의 가상센서를 현재 커서(가장 최근 날짜) 상태로 반환. 실패 시 (None, None)."""
    try:
        from sim.virtual_sensor import available_years
        years = available_years()
    except Exception:
        return None, None
    if not years:
        return None, None
    year = years[-1]
    vs, err = get_vsensor(year)
    if err or vs is None:
        return None, None
    return vs, year


def render_alert_banner(vs):
    """현재 커서 기준 monitor 평가 요약 — 모델·데이터 없으면 조용히 생략."""
    try:
        from llm import expect as expect_mod
        from llm import monitor as monitor_mod
    except Exception:
        return
    expect_model = expect_mod.load_model()
    if expect_model is None:
        return
    try:
        r = vs.reading()
        exp = expect_mod.predict(expect_model, r, vs.date())
        alerts = monitor_mod.assess(r, exp)
    except Exception:
        return
    if not alerts:
        st.success("현재 경보 없음 — 환경이 정상 범위예요.")
        return
    for a in alerts:
        cause_txt = f" · 추정 원인: {a['cause']}" if a.get("cause") else ""
        box = {"경고": st.error, "주의": st.warning}.get(a["level"], st.info)
        box(f"[{a['level']}] {a['reason']}{cause_txt}")


def render_metric_cards(vs):
    """내부온도/습도/CO2/외기 metric 카드 4장 — 다음날 LSTM 예측 delta 포함."""
    try:
        r = vs.reading()
    except Exception:
        unavailable("핵심 지표", "가상 센서 조회 실패")
        return

    delta = None
    try:
        from dl import infer
        live = vs.window()
        fc = infer.forecast(live)
        if fc:
            delta = f"내일 예측 {fc['next_temp']}℃ ({fc['trend']})"
    except Exception:
        delta = None

    metric_row([
        ("내부 온도", f"{r['온도내부_평균']:.1f}℃", delta),
        ("내부 습도", f"{r['습도내부_평균']:.0f}%", None),
        ("CO₂", f"{r['co2_평균']:.0f}", None),
        ("외부 온도", f"{r['온도외부_평균']:.1f}℃", None),
    ])


def render_recent_chart(vs):
    """최근 7일 실측 vs 기대값 미니 차트 — 기대값 모델 없으면 생략."""
    try:
        from dl import infer
        from llm import expect as expect_mod
    except Exception:
        return

    expect_model = expect_mod.load_model()
    if expect_model is None:
        unavailable("실측 vs 기대값 차트", "기대값 모델(models/env_expect_reg.pkl) 없음")
        return

    try:
        import pandas as pd
        live = vs.window()
        win_dates = vs.dates[vs.cursor - infer.WINDOW + 1: vs.cursor + 1]
        rows = []
        for i, d in enumerate(win_dates):
            day_reading = {"온도외부_평균": float(live[i][infer.ENV_FEATURES.index("온도외부_평균")]),
                           "일사량_평균": float(live[i][infer.ENV_FEATURES.index("일사량_평균")])}
            day_exp = expect_mod.predict(expect_model, day_reading, d)
            actual = float(live[i][infer.ENV_FEATURES.index("온도내부_평균")])
            if day_exp is not None:
                rows.append({"날짜": d, "실측": actual, "기대값": day_exp["평균"]})
        if rows:
            st.line_chart(pd.DataFrame(rows).set_index("날짜")[["실측", "기대값"]])
        else:
            unavailable("실측 vs 기대값 차트", "최근 구간 기대값 계산 실패")
    except Exception:
        unavailable("실측 vs 기대값 차트", "차트 생성 중 오류")


def render_shortcuts():
    """기능 바로가기 카드 3장.

    st.page_link는 st.Page 인스턴스를 받아야 streamlit_app.py의 st.navigation 등록과
    매칭된다(파일 경로 문자열을 넘기면 StreamlitPageNotFoundError로 크래시 — 이슈 #10 P1-1).
    nav 모듈은 함수 내부에서 지연 임포트해 dashboard.py 모듈 로드 시점의 순환 참조를 피한다
    (app/nav.py가 이 모듈의 render를 임포트하므로).
    """
    import nav

    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link(nav.PAGE_DIAGNOSIS, label="🔬 잎 병해 진단", icon="🔬")
        st.caption("잎 사진을 업로드해 병해를 진단해요.")
    with c2:
        st.page_link(nav.PAGE_PRESCRIBE, label="💊 AI 처방", icon="💊")
        st.caption("진단 결과로 자연어 처방을 받아요.")
    with c3:
        st.page_link(nav.PAGE_MONITOR, label="🌡️ 환경 관제", icon="🌡️")
        st.caption("가상 센서를 재생하며 설정 밴드 기반 자동제어·경보를 확인해요.")


def render():
    page_header("🏠 농장 대시보드", "오늘의 환경 상태와 경보를 한눈에 확인하세요.")

    vs, year = _latest_vsensor()
    if vs is None:
        unavailable("농장 대시보드 실시간 지표", "가상 센서 데이터(env_daily.csv)·모델 없음",
                     "아래 기능 바로가기에서 각 페이지를 직접 확인하세요")
        render_shortcuts()
        return

    section("경보")
    render_alert_banner(vs)

    section("핵심 지표", f"재생 작기 {year} · 최신 날짜 {vs.date()}")
    render_metric_cards(vs)

    section("최근 7일 실측 vs 기대값")
    render_recent_chart(vs)

    st.divider()
    section("기능 바로가기")
    render_shortcuts()


if __name__ == "__main__":
    st.set_page_config(page_title="농장 대시보드", page_icon="🏠", layout="wide")
    render()
