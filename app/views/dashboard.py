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

이슈 #48 — 핵심 지표를 가상센서 원본이 아니라 "관제 오늘 운영(제어 후)" 값으로 연결한다.
monitor.py의 render_live_tab()과 동일하게 control/live.today_outdoor()(KMA)+
assemble_today_timeline()으로 오늘 제어 후 내부 온·습도·외기를 계산해 KPI 카드에 쓴다.
KMA 키 미설정(로컬/CI)·조회 실패·timeline 없음이면 조용히 가상센서 원본 경로로 폴백한다
(이슈 #10 C4 — 대시보드는 어떤 경우도 크래시하지 않아야 함). CO₂는 KMA에 없으므로
가상센서(vs.reading()) 값을 그대로 쓴다(사용자 확정).

이슈 #51 — 상단 경보 배너도 같은 "관제 제어 후" 값 기준으로 전환. 제어후 값이 밴드 안이면
(=잘 제어되고 있으면) 경보가 자연히 사라지도록 control.live.emergency_hours()(장치
풀가동에도 밴드 밖=제어 한계 초과)로 판정한다. KMA 실패 폴백은 기존 vsensor 원본 기반
monitor.assess() 경로를 그대로 유지.
"""
from datetime import date as _date
from datetime import datetime
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


@st.cache_data(ttl=60)
def _cached_today_outdoor():
    """control.live.today_outdoor() UI 반복 방어층 — monitor.py의 _cached_today_outdoor()와
    동일 이유(이슈 #29 P2)로 대시보드 쪽에도 60s 캐시를 씌운다. rerun마다 동기 KMA 호출이
    반복 블로킹되는 걸 막는다."""
    from control import live as live_mod
    return live_mod.today_outdoor()


def _today_live_timeline():
    """관제 오늘 운영 타임라인(list[dict])+setpoints 전체 — monitor.py의 render_live_tab()과
    동일 방식(assemble_today_timeline)으로 조립한다(이슈 #48, #51).

    대시보드는 세션별 수동 조작(장치 카드 토글)을 반영할 필요가 없으므로 시뮬 상태는 항상
    default_states()(전체 자동)로 고정한다 — render_live_tab()과 달리 세션 K_LIVE_DEVICE_STATES는
    쓰지 않는다.

    KMA 미설정·조회 실패·timeline 없음이면 (None, None)을 반환해 호출측이 가상센서 원본으로
    폴백하게 한다(이슈 #10 C4 — 어떤 경우도 대시보드를 죽이지 않는다). _today_live_kpi()
    (핵심 지표)·render_alert_banner()(경보, 이슈 #51)가 공통으로 재사용한다."""
    try:
        from control import live as live_mod
        from control.actuators import default_states
        from control.setpoints import load as load_setpoints

        outdoor = _cached_today_outdoor()
        if outdoor is None:
            return None, None
        today = _date.today()
        now_hour = datetime.now().hour
        setpoints = load_setpoints()
        timeline = live_mod.assemble_today_timeline(
            outdoor, setpoints, default_states(), today, now_hour)
        if not timeline:
            return None, None
        return timeline, setpoints
    except Exception:
        return None, None


def _today_live_kpi(live_timeline=None):
    """관제 오늘 운영(제어 후) KPI — 현재 시각의 제어 후 내부 온·습도·외기(이슈 #48).

    live_timeline — `_today_live_timeline()`의 (timeline, setpoints) 결과를 이미 조립해
    뒀다면 그대로 넘겨 재사용한다(이슈 #51 리뷰 P2-4 — render()가 경보 배너·KPI 양쪽에
    같은 타임라인을 공유해 assemble_today_timeline()이 render() 1회에 2번 도는 중복
    계산을 없앤다). 생략(None)하면 기존처럼 직접 조회한다(단독 호출·테스트 무회귀).

    KMA 미설정·조회 실패·timeline 없음·필요 값 결측이면 None을 반환해 호출측이 가상센서
    원본으로 폴백하게 한다(이슈 #10 C4 — 어떤 경우도 대시보드를 죽이지 않는다)."""
    try:
        timeline, setpoints = live_timeline if live_timeline is not None else _today_live_timeline()
        if timeline is None:
            return None
        now_hour = datetime.now().hour
        cur = next((t for t in timeline if t["hour"] == now_hour), timeline[-1])
        ctrl_temp, ctrl_hum, out_temp = cur.get("ctrl_temp"), cur.get("ctrl_hum"), cur.get("out_temp")
        if ctrl_temp is None or ctrl_hum is None or out_temp is None:
            return None
        return {"ctrl_temp": ctrl_temp, "ctrl_hum": ctrl_hum, "out_temp": out_temp,
                "today": _date.today(), "setpoints": setpoints}
    except Exception:
        return None


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


def render_alert_banner(vs, live_timeline=None):
    """상단 경보 배너(이슈 #51) — 오늘 운영 "관제 제어 후" 값 기준으로 판정한다.

    핵심 지표(#48)는 이미 제어 후 값을 쓰는데 경보만 가상센서 원본(vs.reading())을 보면
    "고습 93% 경고" 같은 불일치가 생긴다(실제로는 제어가 잘 돼 밴드 안인데 원본 센서는
    여전히 밴드 밖). control.live.emergency_hours()로 현재 시각이 "장치 풀가동에도 밴드
    밖"(제어 한계 초과)인지만 보고, 그 경우에만 "경고"(danger)로 표시한다 — 제어 후 값이
    밴드 안이면(=잘 제어되고 있으면) 경보가 자연히 사라진다. monitor.py 오늘운영 탭의
    "지금" 상태와 동일 기준(emergency_hours)이라 일관성이 유지된다.

    live_timeline — `_today_live_kpi()`와 동일한 (timeline, setpoints) 공유 인자(이슈 #51
    리뷰 P2-4, 중복 계산 방지). 생략하면 직접 조회한다.

    폴백 — KMA 실패로 타임라인이 없으면(제어 후 값 자체가 없음) 기존 vsensor 원본 기반
    monitor.assess() 경로를 그대로 쓴다(로컬/데모에서도 뭔가 보이게, 이슈 #10 C4 무크래시
    원칙과 동일)."""
    try:
        from control import live as live_mod

        timeline, setpoints = live_timeline if live_timeline is not None else _today_live_timeline()
        if timeline is not None:
            now_hour = datetime.now().hour
            emg = live_mod.emergency_hours(timeline, setpoints)
            cur_emg = [e for e in emg if e["hour"] == now_hour]
            if cur_emg:
                for item in cur_emg:
                    alert_strip("danger", item["reason"], severity_label="경고")
            else:
                alert_strip("ok", "현재 경보 없음 — 환경이 정상 범위예요.")
            return
    except Exception:
        pass

    # 폴백(KMA 실패 — 타임라인 없음): 기존 vsensor 원본 기반 assess() 경로.
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


def render_metric_cards(vs, live=None):
    """내부온도/습도/CO2/외기 KPI 카드 4장 — 다음날 LSTM 예측·습도 상한은 상태 칩으로 표현.

    (이슈 #47) 예전엔 "내일 예측 27.6℃"를 st.metric의 delta로 넘겨 초록 상승 화살표로
    오해됐다(항상 초록 ▲로 보여 실제 방향과 무관하게 "좋아지고 있다"는 착시). delta를
    쓰지 않고 온도는 항상 "정상 · 내일 N℃" 칩으로, 습도는 임계 초과 시 "주의"/"경고" 칩으로
    표현해 오독 여지를 없앤다.

    (이슈 #48) 내부 온도·내부 습도·외부 온도는 `live`(_today_live_kpi() 결과)가 있으면
    관제 오늘 운영(제어 후) 값을 쓰고, 없으면(KMA 미설정·실패 — 로컬/CI) 가상센서 원본
    (vs.reading())으로 폴백한다. CO₂는 KMA에 없어 어느 경로든 가상센서 값을 그대로 쓴다.

    (이슈 #55) 습도 칩도 live(제어 후) 경로에선 온도 칩과 대칭으로 밴드(setpoints.hum_low/high)
    기준으로 '주의'를 판정한다 — 경보 배너가 밴드 hum_high 기준(emergency_hours)으로 바뀌면서
    카드 칩(병해 임계 85/90)과 80~85%에서 어긋나던 것을 밴드에 맞춰 일관화. 폴백(가상센서 원본)
    경로는 제어 밴드가 없어 기존 병해 위험 임계(llm.monitor)를 그대로 쓴다."""
    try:
        r = vs.reading()
    except Exception:
        unavailable("핵심 지표", "가상 센서 조회 실패")
        return

    if live is not None:
        temp_val, hum_val, out_temp_val = live["ctrl_temp"], live["ctrl_hum"], live["out_temp"]
    else:
        temp_val, hum_val, out_temp_val = r["온도내부_평균"], r["습도내부_평균"], r["온도외부_평균"]

    temp_chip, temp_level = None, "ok"
    try:
        from dl import infer
        window = vs.window()
        fc = infer.forecast(window)
        if fc:
            if live is not None:
                # 이슈 #48 — 제어 후 값 기준 밴드 재판정. setpoints.temp_low/high 안이면 정상.
                sp = live["setpoints"]
                temp_status = "정상" if sp.temp_low <= temp_val <= sp.temp_high else "주의"
                temp_level = "ok" if temp_status == "정상" else "warn"
            else:
                temp_status = "정상"
            temp_chip = f"{temp_status} · 내일 {fc['next_temp']}℃ ({fc['trend']})"
    except Exception:
        temp_chip = None

    hum = hum_val
    if live is not None:
        # 이슈 #55 — 제어 후 값일 땐 습도 칩도 온도 칩(위 live 분기)과 대칭으로 밴드(setpoints)
        # 기준 판정. 경보 배너가 밴드 hum_high(기본 80) 기준(emergency_hours)으로 바뀌면서,
        # 카드 칩이 병해 임계(85/90)를 쓰면 80~85% 구간에서 배너=경고·카드=정상으로 어긋나던
        # 것을 밴드에 맞춰 일관화한다(danger는 배너가 담당 — 칩은 온도와 동일하게 주의까지).
        sp = live["setpoints"]
        if hum > sp.hum_high or hum < sp.hum_low:
            hum_chip, hum_level = f"주의 · 밴드({sp.hum_low:.0f}~{sp.hum_high:.0f}%) 벗어남", "warn"
        else:
            hum_chip, hum_level = None, "ok"
    else:
        # 폴백(가상센서 원본·데모): 제어 밴드가 없으므로 병해 위험 임계(llm.monitor)로 표시.
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
        {"icon": "🌡", "label": "내부 온도", "value": f"{temp_val:.1f}", "unit": "℃",
         "chip": temp_chip, "chip_level": temp_level},
        {"icon": "💧", "label": "내부 습도", "value": f"{hum:.0f}", "unit": "%",
         "chip": hum_chip, "chip_level": hum_level},
        {"icon": "🫧", "label": "CO₂", "value": f"{r['co2_평균']:.0f}", "unit": "ppm"},
        {"icon": "🌤", "label": "외부 온도", "value": f"{out_temp_val:.1f}", "unit": "℃"},
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

    # 이슈 #51 리뷰 P2-4 — 타임라인을 여기서 한 번만 조립해 경보 배너·핵심 지표 양쪽에
    # 넘긴다(각자 내부에서 다시 조회하면 assemble_today_timeline()이 render() 1회에 2번
    # 돈다). Setpoints/states가 unhashable이라 st.cache_data보다 이 방식이 더 간단하다.
    live_timeline = _today_live_timeline()

    section("경보")
    render_alert_banner(vs, live_timeline)

    live = _today_live_kpi(live_timeline)
    if live is not None:
        section("핵심 지표", f"오늘 {live['today']} · 관제 제어 후 값")
    else:
        section("핵심 지표", f"재생 작기 {year} · 최신 날짜 {vs.date()} · 데모(리플레이) 데이터")
    render_metric_cards(vs, live)

    section("최근 7일 실측 vs 기대값")
    render_recent_chart(vs)

    st.divider()
    section("기능 바로가기")
    render_shortcuts()


if __name__ == "__main__":
    st.set_page_config(page_title="농장 대시보드", page_icon="🏠", layout="wide")
    render()
