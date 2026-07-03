"""
환경 모니터링 페이지 — 가상 센서 재생·기대값 기반 경보·외부 날씨·선견 예보·코치.

구 app/phase3_llm.py 하단부(가상 센서 ~ 조기 경보)를 분리 이관(이슈 #10 C2).
vsensor 생성/시나리오 적용/tick/슬라이더 seek 로직은 문자 그대로 이동(회귀 방지) —
감싸는 것은 state.get_vsensor 뿐, 조건 분기·순서는 원본과 동일하게 유지한다.
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from state import K_LAST_WARNING, K_WEATHER_QA_ANSWER, get_vsensor
from ui import alert_box, metric_row, page_header, section, unavailable


def render_sensor_controls():
    """연도·시나리오 선택 + 날짜 재생(다음 날/슬라이더) → (vs, live, r) 또는 (None, None, None)."""
    from dl import infer

    @st.cache_data(ttl=3600)
    def _years():
        from sim.virtual_sensor import available_years
        return available_years()

    years = _years()
    if not years:
        unavailable("가상 센서", "환경 데이터(env_daily.csv)·LSTM 없음")
        return None, None, None

    from sim.virtual_sensor import SCENARIOS, apply_scenario

    ycol, bcol = st.columns([2, 1])
    with ycol:
        year = st.selectbox("재생 작기(연도 라벨)", years, index=len(years) - 1,
                            help="env_daily.csv의 '연도'는 작기 라벨 — 데이터가 해를 넘길 수 있음")

    vs, err = get_vsensor(year)
    if err:
        st.warning(f"이 작기는 재생할 수 없어요: {err}")
        return None, None, None
    if vs is None:
        return None, None, None

    # 이슈 #6 PR-2 데모 — 정상/한파/히터고장 프리셋으로 원인 구분 경보 시연
    scen_key = f"vsensor_scenario_{year}"
    scenario = st.selectbox("🧪 시뮬 시나리오", SCENARIOS,
                             key=scen_key,
                             help="한파=외기·내부 모두 급락(외기 요인) · 히터고장=외기는 그대로인데 내부만 급락(설비 고장 의심)")
    if st.session_state.get(f"{scen_key}_applied") != scenario:
        apply_scenario(vs, scenario)
        st.session_state[f"{scen_key}_applied"] = scenario

    # 슬라이더 key는 연도별로 분리 — 연도 바뀌면 새 vs.date()로 자연 초기화됨
    date_key = f"vsensor_date_{year}"
    if date_key not in st.session_state:
        st.session_state[date_key] = vs.date()

    def _on_seek_change():
        """슬라이더 이동 → 커서 seek (버튼과 동일한 vs 인스턴스 공유)."""
        vs.seek(st.session_state[date_key])
        apply_scenario(vs, st.session_state[scen_key])   # 이동해도 같은 시나리오 유지

    with bcol:
        st.write("")
        if st.button("다음 날 ▶", use_container_width=True):
            vs.tick()
            apply_scenario(vs, scenario)
            st.session_state[date_key] = vs.date()   # 슬라이더 위젯 상태도 함께 갱신

    st.select_slider(
        "📅 날짜로 이동", options=vs.dates[infer.WINDOW - 1:],
        key=date_key, on_change=_on_seek_change,
    )

    live = vs.window()                       # 이 시점 창을 코치·경보에 명시 전달(전역 상태 X)
    r = vs.reading()
    metric_row([
        ("내부 온도", f"{r['온도내부_평균']:.1f}℃", None),
        ("내부 습도", f"{r['습도내부_평균']:.0f}%", None),
        ("CO₂", f"{r['co2_평균']:.0f}", None),
        ("외부 온도", f"{r['온도외부_평균']:.1f}℃", None),
    ])
    st.caption(f"📅 재생 중인 날짜 — {vs.date()}")
    return vs, live, r


def render_alerts(vs, live, r):
    """기대값(expect.py) 기반 원인 구분 경보 카드."""
    from dl import infer
    from llm import expect as expect_mod
    from llm import monitor as monitor_mod

    expect_model = expect_mod.load_model()
    if expect_model is None:
        unavailable("원인 구분 경보", "기대값 모델(models/env_expect_reg.pkl) 없음", "임계 경보만 표시")
    exp = expect_mod.predict(expect_model, r, vs.date()) if expect_model else None
    alerts = monitor_mod.assess(r, exp)
    if alerts:
        for a in alerts:
            cause_txt = f" · 추정 원인: {a['cause']}" if a.get("cause") else ""
            alert_box(a["level"], f"[{a['level']}] {a['reason']}{cause_txt}")
    else:
        st.caption("정상 범위 — 경보 없음")

    if exp is not None:
        import pandas as pd
        win_dates = vs.dates[vs.cursor - infer.WINDOW + 1: vs.cursor + 1]
        rows = []
        for i, d in enumerate(win_dates):
            day_reading = {"온도외부_평균": float(live[i][infer.ENV_FEATURES.index("온도외부_평균")]),
                           "일사량_평균": float(live[i][infer.ENV_FEATURES.index("일사량_평균")])}
            day_exp = expect_mod.predict(expect_model, day_reading, d)
            actual = float(live[i][infer.ENV_FEATURES.index("온도내부_평균")])
            if day_exp is not None:
                sigma = day_exp["resid_sigma"].get("평균", 0.0)
                rows.append({"날짜": d, "실측": actual, "기대값": day_exp["평균"],
                             "상단(+2σ)": day_exp["평균"] + 2 * sigma,
                             "하단(-2σ)": day_exp["평균"] - 2 * sigma})
        if rows:
            chart_df = pd.DataFrame(rows).set_index("날짜")
            st.caption("최근 7일 실측 vs 기대값(±2σ 밴드) — 밴드 이탈이 클수록 설비 이상 가능성 높아요")
            st.line_chart(chart_df[["실측", "기대값", "상단(+2σ)", "하단(-2σ)"]])

    return expect_model


def render_kma_weather(r, expect_model):
    """외부 날씨(기상청 실황·3일 예보) + 사전 경보."""
    st.caption("가상센서는 과거 특정 연도 재생이고, 아래 외기는 실시간이라 날짜가 다를 수 있어요"
               "(데모 목적 병렬 표시).")

    from llm import weather as kma_weather

    wcol1, wcol2 = st.columns([3, 1])
    with wcol2:
        if st.button("🔄 새로고침", use_container_width=True):
            kma_weather.clear_cache()

    def _v(x, unit=""):
        """관측값 None(누락) → '-' 표시(예: 'None℃' 노출 방지)."""
        return "-" if x is None else f"{x}{unit}"

    try:                                          # 안전망 이중화 — UI가 죽지 않게
        current = kma_weather.get_current()
    except Exception:
        current = {"unavailable": True, "reason": "날씨 조회 중 오류"}
    if current.get("unavailable"):
        unavailable("외부 날씨 조회", current.get("reason", "알 수 없는 오류"), "잠시 후 🔄 새로고침")
    else:
        st.markdown(f"외기 **{_v(current['temp'], '℃')}** · 습도 **{_v(current['humidity'], '%')}** "
                    f"· 강수 **{_v(current['rain'], 'mm')}** (내부 {r['온도내부_평균']:.1f}℃ · "
                    f"습도 {r['습도내부_평균']:.0f}% 와 비교)")

    try:                                          # 안전망 이중화 — UI가 죽지 않게
        fcst = kma_weather.get_forecast_3d()
    except Exception:
        fcst = {"unavailable": True, "reason": "날씨 조회 중 오류"}
    if fcst.get("unavailable"):
        unavailable("3일 예보 조회", fcst.get("reason", "알 수 없는 오류"), "잠시 후 🔄 새로고침")
    else:
        daily = fcst.get("daily") or []
        if daily:
            st.dataframe(
                [{"날짜": d["date"], "최저(℃)": d["tmn"], "최고(℃)": d["tmx"]} for d in daily],
                hide_index=True, use_container_width=True,
            )
        hourly = fcst.get("hourly") or []
        if hourly:
            import pandas as pd
            chart_df = pd.DataFrame(hourly)[["date", "time", "temp"]]
            chart_df["시각"] = chart_df["date"] + " " + chart_df["time"]
            st.line_chart(chart_df.set_index("시각")["temp"])

        if expect_model is not None:
            from llm import monitor as ff_monitor
            ff = ff_monitor.feedforward_alerts(fcst.get("daily") or [], expect_model=expect_model)
            if ff:
                for a in ff:
                    st.warning(f"🔮 사전 경보(실시간 예보) — {a['reason']}")


def render_replay_forecast(vs, expect_model):
    """리플레이 선견 예보 — 가상센서 커서+1~3일 실제 외기를 예보처럼 사용."""
    from dl import infer

    if expect_model is None or vs.cursor + 1 >= len(vs.series):
        st.caption("리플레이 선견 구간 없음(기대값 모델 없음 또는 재생 마지막 구간)")
        return
    st.caption("가상센서가 재생 중인 미래 1~3일 실외기를 KMA 예보처럼 흉내낸 데모용 사전 경보예요"
               "(실제 미래를 미리 아는 리플레이 특유 트릭 — 날짜 불일치 방지).")
    replay_daily = []
    for k in range(1, 4):
        idx = vs.cursor + k
        if idx >= len(vs.series):
            break
        outer = float(vs.series[idx][infer.ENV_FEATURES.index("온도외부_평균")])
        replay_daily.append({"date": vs.dates[idx], "tmn": outer - 2.0, "tmx": outer + 2.0})
    from llm import monitor as replay_monitor
    replay_ff = replay_monitor.feedforward_alerts(replay_daily, expect_model=expect_model)
    if replay_ff:
        for a in replay_ff:
            st.warning(f"🔮 리플레이 사전 경보 — {a['reason']}")
    else:
        st.caption("리플레이 선견 구간 — 사전 경보 없음(정상 범위)")


def render_weather_qa():
    section("💬 날씨 질문")
    weather_q = st.text_input("궁금한 점을 물어보세요", value="",
                               placeholder="내일 밤 기온 괜찮을까?", key="weather_qa_input")
    if st.button("날씨 질문", key="weather_qa_btn"):
        from llm import pipeline as llm_pipeline
        with st.spinner("날씨 확인 중 (모델·환경에 따라 수십 초 걸려요)"):
            st.session_state[K_WEATHER_QA_ANSWER] = llm_pipeline.weather_qa(weather_q or "오늘 날씨 어때?")
    if K_WEATHER_QA_ANSWER in st.session_state:
        st.markdown(st.session_state[K_WEATHER_QA_ANSWER])


def render_coach(live):
    from dl import infer

    fc = infer.forecast(live)
    if fc:
        st.markdown(f"**다음날 예측** {fc['next_temp']}℃ ({fc['trend']}) · "
                    f"**습도위험** {fc['humidity_risk']} (최근 7일 평균 {fc['humidity_mean']}%)")

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("🌅 오늘의 코치", use_container_width=True):
            from llm import pipeline
            with st.spinner("코칭 생성 중 (모델·환경에 따라 수십 초 걸려요)"):
                coach = pipeline.daily_coach(live)
            st.success(coach.요약)
            for todo in coach.오늘_할일:
                st.markdown(f"- {todo}")
            st.caption(f"근거: {coach.근거}")
    with cc2:
        if st.button("⚠️ 조기 경보", use_container_width=True):
            from llm import pipeline
            with st.spinner("경보 판단 중 (모델·환경에 따라 수십 초 걸려요)"):
                st.session_state[K_LAST_WARNING] = pipeline.early_warning(live)

    if K_LAST_WARNING in st.session_state:
        from llm import notify
        w = st.session_state[K_LAST_WARNING]
        level = w.경보수준 if w.경보수준 in ("경고", "주의") else "정보"
        alert_box(level, f"경보수준: {w.경보수준}" + (f" · 위험병해: {w.위험병해}" if w.위험병해 and w.위험병해 != "없음" else ""))
        st.markdown(f"- **이유** — {w.이유}")
        if w.권장조치:
            st.markdown(f"- **권장조치** — {w.권장조치}")
        if st.button("📣 디스코드로 보내기", key="send_warn"):
            ok, msg = notify.notify_warning(w)
            (st.success if ok else st.warning)(msg)


def render():
    page_header("🌡️ 환경 모니터링", "실 센서 대신 특정 연도 토마토 온실 실데이터를 하루씩 재생 → 최근 7일로 다음날 예측.")

    vs, live, r = render_sensor_controls()
    if vs is None:
        return

    section("경보")
    expect_model = render_alerts(vs, live, r)

    tab_expect, tab_kma, tab_replay = st.tabs(["기대값 vs 실측", "외부 날씨(KMA)", "선견 예보(리플레이)"])
    with tab_expect:
        st.caption("경보 카드에 포함된 밴드 차트를 참고하세요(위 «경보» 섹션).")
    with tab_kma:
        render_kma_weather(r, expect_model)
        render_weather_qa()
    with tab_replay:
        render_replay_forecast(vs, expect_model)

    st.divider()
    section("코치 · 조기 경보")
    render_coach(live)

    st.divider()
    st.caption("환각 방어 3종: 신뢰도 톤 분기 · 게이트 차단 안내 · 클래스 한정성(잎 병해 4종).")


if __name__ == "__main__":
    st.set_page_config(page_title="환경 모니터링", page_icon="🌡️", layout="wide")
    render()
