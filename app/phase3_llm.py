"""
Phase 3 (LLM) — 자연어 처방 페이지 (3-1: Ollama function calling)

잎 사진 → DL 진단(라벨·확률, 근거) + LLM 처방(초보자 눈높이 자연어).
분업: 진단=DL(resnet18·게이트·YOLO) / 설명·처방=LLM(Ollama, 모델은 .env OLLAMA_MODEL — 로컬 14b·서버 7b).
멀티페이지: app/streamlit_app.py 가 render() 를 호출(set_page_config 는 엔트리에서 1회).

실행:  streamlit run app/streamlit_app.py   (프로젝트 루트에서)
전제:  Ollama 데몬 구동 + `ollama pull <OLLAMA_MODEL>` + `ollama pull bge-m3`
"""
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env", override=True)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")   # 표시용 — 실제 사용 모델과 같은 소스(.env)
SAMPLES = ROOT / "app" / "samples"
SAMPLE_KR = {"late_blight": "🦠 잎마름역병", "leaf_mold": "🦠 잎곰팡이병",
             "normal": "🌿 정상", "tylcv": "🦠 황화잎말이바이러스"}


def _resolve_image(uploaded, sample_key):
    """업로드 파일 or 샘플 → 로컬 파일 경로(추론 tool 은 경로를 받는다)."""
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        return tmp.name
    if sample_key:
        p = SAMPLES / f"{sample_key}.jpg"
        return str(p) if p.exists() else None
    return None


def render():
    st.title("💬 Phase 3 · LLM — 자연어 처방")
    st.caption(f"DL 진단(라벨·확률)을 받아 LLM(Ollama · {MODEL})이 초보자 눈높이 처방으로. "
               "**진단=DL, 설명·처방=LLM** (LLM은 진단하지 않음).")

    # Ollama 구동 확인
    import ollama
    try:
        ollama.list()
    except Exception:
        st.error("⚠️ Ollama 데몬이 실행 중이 아니에요. Ollama 앱을 켜거나 터미널에서 `ollama serve` 후 "
                 f"`ollama pull {MODEL}` 을 실행해 주세요.")
        return

    st.markdown("#### 1) 잎 사진")
    c1, c2 = st.columns(2)
    with c1:
        uploaded = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"])
    with c2:
        choice = st.radio("또는 샘플 선택", ["(선택 안 함)", *SAMPLE_KR.keys()],
                          format_func=lambda k: SAMPLE_KR.get(k, k), horizontal=False)
    sample_key = choice if choice in SAMPLE_KR else None

    # 미리보기 (경로 만들지 않고 위젯 값으로 바로)
    if uploaded is not None:
        st.image(uploaded, caption="입력 잎 사진", width=300)
    elif sample_key:
        st.image(str(SAMPLES / f"{sample_key}.jpg"), caption=f"샘플 · {SAMPLE_KR[sample_key]}", width=300)

    question = st.text_input("2) 물어볼 말",
                             value="이 토마토 잎 좀 봐줘. 병이면 어떻게 조치해야 해?")

    has_image = uploaded is not None or sample_key is not None
    if st.button("💊 처방 받기", type="primary", disabled=not has_image):
        image_path = _resolve_image(uploaded, sample_key)
        if not image_path:
            st.error("이미지를 찾을 수 없어요.")
        else:
            from llm import tools
            from llm.prescribe import prescribe
            with st.spinner(f"DL 진단 + LLM 처방 생성 중… ({MODEL} — 환경에 따라 수십 초~2분 걸릴 수 있어요) "
                             f"(생성 중 다른 버튼을 누르면 취소돼요)"):
                st.session_state["last_diag"] = tools.get_diagnosis(image_path)
                st.session_state["last_presc"] = prescribe(question, image_path=image_path)

    # 처방 결과 렌더(세션 보관 — rerun/전송 버튼에도 유지)
    if "last_presc" in st.session_state:
        from dl import infer
        from llm import notify
        diag = st.session_state["last_diag"]
        presc = st.session_state["last_presc"]

        st.markdown("#### 🔬 DL 진단 (근거)")
        if diag.get("ood_blocked"):
            st.warning(f"진단 차단 — {diag.get('reason')}. 병명을 단정하지 않고 재촬영을 안내합니다.")
        else:
            st.markdown(f"**진단:** {diag['label_kr']} · **신뢰도:** {diag['prob'] * 100:.0f}%")
            st.progress(min(max(diag["prob"], 0.0), 1.0))
            st.caption("클래스별 확률 · " + "  ".join(
                f"{infer.LABEL_KR[k]} {v * 100:.0f}%" for k, v in diag["probs"].items()))

        st.markdown("#### 💬 LLM 처방")
        st.success(f"**{presc.진단요약}**")
        st.markdown(
            f"- **원인** — {presc.원인}\n"
            f"- **즉시 조치** — {presc.즉시조치}\n"
            f"- **예방** — {presc.예방}\n"
            f"- **재촬영 시점** — {presc.재촬영시점}"
        )
        if presc.근거출처:
            st.markdown("**📖 근거 출처 (농사로/NCPMS RAG)**")
            for s in presc.근거출처:
                st.markdown(f"- {s}")
        else:
            st.caption("ℹ️ 이 진단에 대한 재배가이드 근거를 찾지 못했어요.")

        if st.button("📣 디스코드로 보내기", key="send_presc"):
            ok, msg = notify.notify_prescription(presc)
            (st.success if ok else st.warning)(msg)

    # ── 가상 센서 · 환경 예측 · 코치 · 경보 (3-3) ──
    st.divider()
    st.subheader("🌡️ 가상 센서 · 환경 예측 (LSTM · 특정 연도 재생)")
    st.caption("실 센서 대신 특정 연도 토마토 온실 실데이터를 하루씩 재생 → 최근 7일로 다음날 예측.")

    from dl import infer

    @st.cache_data(ttl=3600)
    def _years():
        from sim.virtual_sensor import available_years
        return available_years()

    years = _years()
    if not years:
        st.caption("환경 데이터(env_daily.csv)·LSTM 없음 — 가상 센서 비활성")
    else:
        from sim.virtual_sensor import VirtualSensor
        ycol, bcol = st.columns([2, 1])
        with ycol:
            year = st.selectbox("재생 작기(연도 라벨)", years, index=len(years) - 1,
                                help="env_daily.csv의 '연도'는 작기 라벨 — 데이터가 해를 넘길 수 있음")
        vs = st.session_state.get("vsensor")
        if vs is None or vs.year != year:
            try:
                vs = VirtualSensor(year)
                st.session_state["vsensor"] = vs
            except ValueError as e:                 # 그 작기에 7일↑ 시계열 없음
                st.warning(f"이 작기는 재생할 수 없어요: {e}")
                vs = None
        if vs is not None:
            from sim.virtual_sensor import SCENARIOS, apply_scenario

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
            st.markdown(f"📅 **{vs.date()}** · 내부 {r['온도내부_평균']:.1f}℃ · 습도 {r['습도내부_평균']:.0f}% "
                        f"· CO₂ {r['co2_평균']:.0f} · 외부 {r['온도외부_평균']:.1f}℃")

            # ── 🔎 원인 구분 경보 (expect.py 기대값 vs 실측) ──────────────────
            from llm import expect as expect_mod
            from llm import monitor as monitor_mod

            expect_model = expect_mod.load_model()
            if expect_model is None:
                st.caption("ℹ️ 기대값 모델(models/env_expect_reg.pkl) 없음 — 원인 구분 비활성(임계 경보만).")
            exp = expect_mod.predict(expect_model, r, vs.date()) if expect_model else None
            alerts = monitor_mod.assess(r, exp)
            if alerts:
                for a in alerts:
                    box = {"경고": st.error, "주의": st.warning}.get(a["level"], st.info)
                    cause_txt = f" · 추정 원인: {a['cause']}" if a.get("cause") else ""
                    box(f"[{a['level']}] {a['reason']}{cause_txt}")
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
                    st.caption("최근 7일 실측 vs 기대값(±2σ 밴드) — 밴드 이탈이 클수록 설비 이상 가능성↑")
                    st.line_chart(chart_df[["실측", "기대값", "상단(+2σ)", "하단(-2σ)"]])

            # ── 🌤 외부 날씨 (기상청 실황) — 가상센서(과거 replay)와 실시간 외기를 나란히 대비 ──
            st.markdown("##### 🌤 외부 날씨 (기상청 실황)")
            st.caption("⚠️ 가상센서는 과거 특정 연도 재생(위 날짜)이고, 아래 외기는 실시간이라 날짜가 다를 수 있어요"
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
                st.caption(f"ℹ️ 외부 날씨 조회 불가 — {current.get('reason', '알 수 없는 오류')} "
                           "(기상청 응답 지연/제한일 수 있음 — 잠시 후 🔄 새로고침)")
            else:
                st.markdown(f"외기 **{_v(current['temp'], '℃')}** · 습도 **{_v(current['humidity'], '%')}** "
                            f"· 강수 **{_v(current['rain'], 'mm')}** (내부 {r['온도내부_평균']:.1f}℃ · "
                            f"습도 {r['습도내부_평균']:.0f}% 와 비교)")

            try:                                          # 안전망 이중화 — UI가 죽지 않게
                fcst = kma_weather.get_forecast_3d()
            except Exception:
                fcst = {"unavailable": True, "reason": "날씨 조회 중 오류"}
            if fcst.get("unavailable"):
                st.caption(f"ℹ️ 3일 예보 조회 불가 — {fcst.get('reason', '알 수 없는 오류')} "
                           "(기상청 응답 지연/제한일 수 있음 — 잠시 후 🔄 새로고침)")
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

            # ── 🔮 리플레이 선견 예보 — 가상센서 커서+1~3일 실제 외기를 예보처럼 사용 ──
            if expect_model is not None and vs.cursor + 1 < len(vs.series):
                st.markdown("##### 🔮 리플레이 선견 예보 (가상센서, 실 예보 아님)")
                st.caption("가상센서가 재생 중인 미래 1~3일 실외기를 KMA 예보처럼 흉내낸 데모용 사전 경보입니다"
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

            st.markdown("###### 💬 날씨 질문")
            weather_q = st.text_input("궁금한 점을 물어보세요", value="",
                                       placeholder="내일 밤 기온 괜찮을까?", key="weather_qa_input")
            if st.button("날씨 질문", key="weather_qa_btn"):
                from llm import pipeline as llm_pipeline
                with st.spinner("날씨 확인 중…"):
                    st.session_state["weather_qa_answer"] = llm_pipeline.weather_qa(weather_q or "오늘 날씨 어때?")
            if "weather_qa_answer" in st.session_state:
                st.markdown(st.session_state["weather_qa_answer"])

            fc = infer.forecast(live)
            if fc:
                st.markdown(f"**→ 다음날 예측** {fc['next_temp']}℃ ({fc['trend']}) · "
                            f"**습도위험** {fc['humidity_risk']} (최근 7일 평균 {fc['humidity_mean']}%)")

            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("🌅 오늘의 코치", use_container_width=True):
                    from llm import pipeline
                    with st.spinner("코칭 생성 중…"):
                        coach = pipeline.daily_coach(live)
                    st.success(coach.요약)
                    for todo in coach.오늘_할일:
                        st.markdown(f"- {todo}")
                    st.caption(f"근거: {coach.근거}")
            with cc2:
                if st.button("⚠️ 조기 경보", use_container_width=True):
                    from llm import pipeline
                    with st.spinner("경보 판단 중…"):
                        st.session_state["last_warning"] = pipeline.early_warning(live)

            if "last_warning" in st.session_state:
                from llm import notify
                w = st.session_state["last_warning"]
                box = {"경고": st.error, "주의": st.warning}.get(w.경보수준, st.info)
                box(f"경보수준: {w.경보수준}" + (f" · 위험병해: {w.위험병해}" if w.위험병해 and w.위험병해 != "없음" else ""))
                st.markdown(f"- **이유** — {w.이유}")
                if w.권장조치:
                    st.markdown(f"- **권장조치** — {w.권장조치}")
                if st.button("📣 디스코드로 보내기", key="send_warn"):
                    ok, msg = notify.notify_warning(w)
                    (st.success if ok else st.warning)(msg)

    st.divider()
    st.caption("환각 방어 3종: ① 신뢰도 톤 분기 · ② 게이트 차단 안내 · ③ 클래스 한정성(잎 병해 4종). "
               "진행 상황 → `docs/roadmap.md` Phase 3.")


if __name__ == "__main__":
    st.set_page_config(page_title="Phase 3 · LLM", page_icon="💬", layout="wide")
    render()
