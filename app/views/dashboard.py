"""
농장 대시보드 페이지 — [서비스] 그룹 기본(default) 페이지.

상단 경보 배너 + 핵심 지표 카드 4장 + 최근 7일 실측 vs 기대값 미니 차트 + 기능 바로가기 카드.
모델·데이터가 없는 환경에서도 절대 죽지 않도록 모든 데이터 접근을 try/exists 가드로 감싼다(이슈 #10 C4).

이슈 #47 — 야외 가독 미니멀 리디자인: metric_row(st.metric delta) 대신 kpi_cards(HTML 카드) +
상태 칩으로 전환. 기존엔 "내일 예측 27.6℃"를 delta로 넘겨 초록 상승 화살표로 오해되던 문제를
없애고, 예측/상태는 항상 칩으로 표현한다. 차트는 st.line_chart 대신 Altair로 y축을 데이터
범위(±8% 여유)로 확대해 변화가 눈에 띄게 한다(monitor.py의 _live_trend_chart 패턴 참고).

딥그린 다크 기본/라이트 토글 확장(이슈 #47 2차) — 차트 색은 ui.current_theme()에 따라
다크(밝은 그린 #5FD08A·짙은 배경 그리드)/라이트(진한 그린 #3F7D23·옅은 그리드)로 갈아 낀다.
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from state import get_vsensor
from ui import alert_strip, current_theme, kpi_cards, page_header, section, unavailable

# 습도 KPI 칩 임계값 — llm.monitor.assess()의 경보 임계와 동일 소스(이슈 #47, 중복 정의 방지)
_HUM_WARN_FALLBACK, _HUM_CRIT_FALLBACK = 85.0, 90.0

# 최근 7일 차트 팔레트(이슈 #47 2차) — ui.py의 _DARK/_LIGHT KPI 팔레트와 톤을 맞춤
_CHART_PALETTE = {
    "dark": {"accent": "#5FD08A", "bg": "#0F1C15", "grid": "#1C2C22", "axis_text": "#6F8577"},
    "light": {"accent": "#3F7D23", "bg": "#FFFFFF", "grid": "#EEF1EA", "axis_text": "#9AA891"},
}


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
    """현재 커서 기준 monitor 평가 요약 — 모델·데이터 없으면 조용히 생략.

    경보 레벨(한국어 "경고"/"주의")을 alert_strip의 'danger'/'warn'으로 매핑한다."""
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
        alert_strip("ok", "현재 경보 없음 — 환경이 정상 범위예요.")
        return
    level_map = {"경고": "danger", "주의": "warn"}
    for a in alerts:
        cause_txt = f" · 추정 원인: {a['cause']}" if a.get("cause") else ""
        alert_strip(level_map.get(a["level"], "warn"), f"{a['reason']}{cause_txt}",
                     severity_label=a["level"])


def render_metric_cards(vs):
    """내부온도/습도/CO2/외기 KPI 카드 4장 — 다음날 LSTM 예측·습도 상한은 상태 칩으로 표현.

    (이슈 #47) 예전엔 "내일 예측 27.6℃"를 st.metric의 delta로 넘겨 초록 상승 화살표로
    오해됐다(항상 초록 ▲로 보여 실제 방향과 무관하게 "좋아지고 있다"는 착시). delta를
    쓰지 않고 온도는 항상 "정상 · 내일 N℃" 칩으로, 습도는 임계 초과 시 "주의"/"경고" 칩으로
    표현해 오독 여지를 없앤다."""
    try:
        r = vs.reading()
    except Exception:
        unavailable("핵심 지표", "가상 센서 조회 실패")
        return

    temp_chip, temp_level = None, "ok"
    try:
        from dl import infer
        live = vs.window()
        fc = infer.forecast(live)
        if fc:
            temp_chip = f"정상 · 내일 {fc['next_temp']}℃ ({fc['trend']})"
    except Exception:
        temp_chip = None

    hum = r["습도내부_평균"]
    hum_warn, hum_crit = _HUM_WARN_FALLBACK, _HUM_CRIT_FALLBACK
    try:
        from llm import monitor as monitor_mod
        hum_warn, hum_crit = monitor_mod.HUM_WARN, monitor_mod.HUM_CRIT
    except Exception:
        pass
    if hum >= hum_crit:
        hum_chip, hum_level = f"경고 · 상한 {hum_crit:.0f}% 초과", "danger"
    elif hum >= hum_warn:
        hum_chip, hum_level = f"주의 · 상한 {hum_warn:.0f}% 초과", "warn"
    else:
        hum_chip, hum_level = None, "ok"

    kpi_cards([
        {"icon": "🌡", "label": "내부 온도", "value": f"{r['온도내부_평균']:.1f}", "unit": "℃",
         "chip": temp_chip, "chip_level": temp_level},
        {"icon": "💧", "label": "내부 습도", "value": f"{hum:.0f}", "unit": "%",
         "chip": hum_chip, "chip_level": hum_level},
        {"icon": "🫧", "label": "CO₂", "value": f"{r['co2_평균']:.0f}", "unit": "ppm"},
        {"icon": "🌤", "label": "외부 온도", "value": f"{r['온도외부_평균']:.1f}", "unit": "℃"},
    ])


def _recent_trend_chart(rows: list[dict]):
    """최근 7일 실측 vs 기대값 Altair 차트(이슈 #47) — y축을 데이터 범위(min/max ±8% 여유)로
    확대(0부터 시작 금지), 실측=실선+area fill(끝점만 테두리 마커), 기대값=점선. monitor.py의
    _live_trend_chart y축 확대 패턴을 참고했다. 옅은 가로 그리드 + 색은 현재 테마(다크/라이트)에
    맞춰 갈아 낀다(ui.current_theme())."""
    import altair as alt
    import pandas as pd

    pal = _CHART_PALETTE.get(current_theme(), _CHART_PALETTE["dark"])

    df = pd.DataFrame(rows)
    vals = pd.concat([df["실측"], df["기대값"]]).dropna()
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.08 or 1.0
    y_scale = alt.Scale(domain=[lo - pad, hi + pad], zero=False, nice=False, clamp=True)

    x_enc = alt.X("날짜:N", title=None, sort=None)
    area = alt.Chart(df).mark_area(color=pal["accent"], opacity=0.22).encode(
        x=x_enc, y=alt.Y("실측:Q", title=None, scale=y_scale),
    )
    expect_line = alt.Chart(df).mark_line(
        color=pal["accent"], strokeWidth=1.6, strokeDash=[5, 4], opacity=0.65, point=False,
    ).encode(x=x_enc, y=alt.Y("기대값:Q", title=None, scale=y_scale))
    actual_line = alt.Chart(df).mark_line(
        color=pal["accent"], strokeWidth=2.4, point=False,
    ).encode(x=x_enc, y=alt.Y("실측:Q", title=None, scale=y_scale))
    end_point = alt.Chart(df.iloc[[-1]]).mark_circle(
        size=90, color=pal["accent"], stroke=pal["bg"], strokeWidth=2,
    ).encode(x=x_enc, y=alt.Y("실측:Q", title=None, scale=y_scale))

    # 이슈 #47 2차 — Streamlit은 config.toml의 base(다크 고정)를 따라 Vega 차트에도 자체 다크
    # 테마를 입혀서, 라이트 토글 시에도 차트 배경만 그대로 어둡게 남는 버그가 있었다. 차트
    # 배경(configure background)을 현재 팔레트로 명시해 항상 앱 배경과 일치시킨다.
    return (area + expect_line + actual_line + end_point).properties(height=220).configure(
        background=pal["bg"]
    ).configure_view(
        strokeWidth=0, fill=pal["bg"],
    ).configure_axisY(
        gridColor=pal["grid"], gridOpacity=1, domain=False, ticks=False,
        labelColor=pal["axis_text"], labelFontSize=9, tickCount=4,
    ).configure_axisX(
        domain=False, ticks=False, labelColor=pal["axis_text"], labelFontSize=10,
    )


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
            st.altair_chart(_recent_trend_chart(rows), use_container_width=True)
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

    (이슈 #47) 클릭 라우팅이 깨지면 안 되므로 st.page_link는 그대로 유지하고,
    st.container(border=True, key="sf_shortcut_*")로 감싸 카드처럼 보이게만 CSS 스타일링한다
    (완전 커스텀 HTML <a>로 대체하지 않음 — ui.py의 .sf-shortcut-icon/div[class*=st-key-sf_shortcut]
    CSS 참조). 아이콘은 그린 그라데이션 원(장식용 HTML)을 page_link 위에 별도로 얹는다."""
    import nav

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True, key="sf_shortcut_1"):
        st.markdown('<div class="sf-shortcut-icon">🔬</div>', unsafe_allow_html=True)
        st.page_link(nav.PAGE_DIAGNOSIS, label="잎 병해 진단")
        st.caption("잎 사진을 업로드해 병해를 진단해요.")
    with c2, st.container(border=True, key="sf_shortcut_2"):
        st.markdown('<div class="sf-shortcut-icon">💊</div>', unsafe_allow_html=True)
        st.page_link(nav.PAGE_PRESCRIBE, label="AI 처방")
        st.caption("진단 결과로 자연어 처방을 받아요.")
    with c3, st.container(border=True, key="sf_shortcut_3"):
        st.markdown('<div class="sf-shortcut-icon">🌡️</div>', unsafe_allow_html=True)
        st.page_link(nav.PAGE_MONITOR, label="환경 관제")
        st.caption("가상 센서를 재생하며 설정 밴드 기반 자동제어·경보를 확인해요.")


def render():
    page_header("🏠 농장 대시보드", "오늘의 환경 상태와 경보를 한눈에 확인하세요.", badge="데모")

    vs, year = _latest_vsensor()
    if vs is None:
        unavailable("농장 대시보드 실시간 지표", "가상 센서 데이터(env_daily.csv)·모델 없음",
                     "아래 기능 바로가기에서 각 페이지를 직접 확인하세요")
        render_shortcuts()
        return

    section("경보")
    render_alert_banner(vs)

    section("핵심 지표", f"재생 작기 {year} · 최신 날짜 {vs.date()} · 데모(리플레이) 데이터")
    render_metric_cards(vs)

    section("최근 7일 실측 vs 기대값")
    render_recent_chart(vs)

    st.divider()
    section("기능 바로가기")
    render_shortcuts()


if __name__ == "__main__":
    st.set_page_config(page_title="농장 대시보드", page_icon="🏠", layout="wide")
    render()
