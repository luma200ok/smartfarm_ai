"""환경 관제 페이지 — 가상 센서 재생·설정 밴드 기반 장치 자동제어(시뮬)·제어 로그·긴급 알림.

구 app/phase3_llm.py 하단부(가상 센서 ~ 조기 경보)를 분리 이관(이슈 #10 C2) 이후,
관제형 대시보드로 전면 개편(이슈 #17) — 규칙 기반 자동 제어(src/control/)와 제어 로그·
긴급 경보를 중심으로 재구성한다. vsensor 생성/시나리오 적용/tick/슬라이더 seek 로직은
문자 그대로 유지(회귀 방지) — 감싸는 것은 state.get_vsensor 뿐, 조건 분기·순서는 그대로.
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from state import (K_CONTROL_ACTIVE, K_CONTROL_LAST_DATE, K_CONTROL_LOG,
                    K_DEVICE_STATES, K_DISCORD_CONTROL_TOGGLE, K_RECENT_READINGS,
                    K_SETPOINTS, get_vsensor)
from ui import alert_box, metric_row, page_header, section, unavailable

_MAX_RECENT = 10       # emergency() 판정에 필요한 3틱보다 여유 있게 보관
_MAX_LOG_ROWS = 30      # 제어 로그 테이블 최근 N건만 표시


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


def _get_setpoints():
    from control.setpoints import Setpoints
    if K_SETPOINTS not in st.session_state:
        st.session_state[K_SETPOINTS] = Setpoints()
    return st.session_state[K_SETPOINTS]


def _get_device_states():
    from control.actuators import default_states
    if K_DEVICE_STATES not in st.session_state:
        st.session_state[K_DEVICE_STATES] = default_states()
    return st.session_state[K_DEVICE_STATES]


def _control_embed(logs: list, date) -> dict:
    from control.actuators import DEVICE_LABEL_KR
    from llm import notify
    lines = [f"{DEVICE_LABEL_KR.get(log.device, log.device)} {log.action} — {log.reason}"
             for log in logs]
    fields = [
        {"name": "일시", "value": str(date), "inline": True},
        {"name": "제어 내역", "value": notify._t("\n".join(lines))},
    ]
    return {"title": "🎛 자동 제어 이벤트", "color": 3447003, "fields": fields}


def _emergency_embed(alert: dict, date) -> dict:
    from llm import notify
    fields = [
        {"name": "일시", "value": str(date), "inline": True},
        {"name": "레벨", "value": alert["level"], "inline": True},
        {"name": "사유", "value": notify._t(alert["reason"])},
    ]
    return {"title": "🚨 제어 한계 초과 — 긴급", "color": 15158332, "fields": fields}


def _run_control_step(vs, r, setpoints, states):
    """현재 날짜가 이전 처리분과 다를 때만 1회 — decide()→효과 반영→긴급 판정→(옵션)디스코드."""
    from control import controller
    from control.effects import apply_effects
    from llm import notify

    year_date = (vs.year, vs.date())
    last = st.session_state.get(K_CONTROL_LAST_DATE)
    log_list = st.session_state.setdefault(K_CONTROL_LOG, [])
    active = st.session_state.setdefault(K_CONTROL_ACTIVE, set())
    recent = st.session_state.setdefault(K_RECENT_READINGS, [])

    if last == year_date:
        return log_list

    logs = controller.decide(r, setpoints, states, date=vs.date())
    if logs:
        log_list.extend(logs)
        apply_effects(vs, states, start=vs.cursor + 1, days=1)
        if st.session_state.get(K_DISCORD_CONTROL_TOGGLE):
            ok, msg = notify.send_discord(_control_embed(logs, vs.date()))
            if not ok:
                st.session_state["_control_discord_msg"] = msg

    recent.append(r)
    if len(recent) > _MAX_RECENT:
        del recent[: len(recent) - _MAX_RECENT]

    alert = controller.emergency(recent, setpoints, states, active)
    if alert is not None:
        active.add(controller.akey(alert))
        ok, msg = notify.send_discord(_emergency_embed(alert, vs.date()))
        log_list.append({"date": vs.date(), "device": "-", "action": "긴급",
                          "reason": alert["reason"], "mode": "system"})
        st.session_state["_last_emergency"] = {**alert, "date": vs.date(), "sent": ok, "msg": msg}

    st.session_state[K_CONTROL_LAST_DATE] = year_date
    return log_list


def render_setpoints(setpoints):
    section("⚙️ 설정 밴드", "히스테리시스 데드밴드로 장치가 경계에서 깜빡이지 않게 해요.")
    c1, c2 = st.columns(2)
    with c1:
        setpoints.temp_low, setpoints.temp_high = st.slider(
            "온도 밴드(℃)", 0.0, 40.0, (setpoints.temp_low, setpoints.temp_high), 0.5)
    with c2:
        setpoints.hum_low, setpoints.hum_high = st.slider(
            "습도 밴드(%)", 0.0, 100.0, (setpoints.hum_low, setpoints.hum_high), 1.0)


def render_devices(states, setpoints, r, date):
    from control import controller
    from control.actuators import DEVICE_LABEL_KR, DEVICES

    section("🔌 장치", "자동 모드는 설정 밴드에 따라 자동으로 켜지고 꺼져요. 수동으로 직접 제어할 수도 있어요.")
    cols = st.columns(4)
    for col, device in zip(cols, DEVICES):
        state = states[device]
        with col:
            st.markdown(f"**{DEVICE_LABEL_KR[device]}**")
            status_badge = "🟢 ON" if state.on else "⚪ OFF"
            st.caption(status_badge + (" · 자동" if state.auto else " · 수동"))
            new_auto = st.toggle("자동 모드", value=state.auto, key=f"auto_{device}")
            if new_auto != state.auto:
                state.auto = new_auto
            if not state.auto:
                if st.button(("끄기" if state.on else "켜기"), key=f"manual_{device}",
                              use_container_width=True):
                    state.on = not state.on
                    log_list = st.session_state.setdefault(K_CONTROL_LOG, [])
                    log_list.append(controller.ControlLog(
                        date=str(date), device=device,
                        action="ON" if state.on else "OFF",
                        reason="수동 조작", mode="manual"))


def render_trend(vs, live, setpoints):
    """최근 WINDOW일 추이(live 창, read-time overlay 반영) + 밴드 상/하한 오버레이."""
    from dl import infer
    import pandas as pd

    section("📈 온·습도 추이", "최근 재생 구간 — 밴드를 벗어난 구간에서 장치가 자동 대응해요.")
    win_dates = vs.dates[vs.cursor - infer.WINDOW + 1: vs.cursor + 1]
    temp_i = infer.ENV_FEATURES.index("온도내부_평균")
    hum_i = infer.ENV_FEATURES.index("습도내부_평균")
    rows = [{"날짜": d, "온도": float(live[i][temp_i]), "습도": float(live[i][hum_i])}
            for i, d in enumerate(win_dates)]
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("날짜")
    df["온도상한"] = setpoints.temp_high
    df["온도하한"] = setpoints.temp_low
    st.line_chart(df[["온도", "온도상한", "온도하한"]])
    df2 = pd.DataFrame(rows).set_index("날짜")
    df2["습도상한"] = setpoints.hum_high
    df2["습도하한"] = setpoints.hum_low
    st.line_chart(df2[["습도", "습도상한", "습도하한"]])


def render_control_log():
    section("🧾 제어 로그", "장치 ON/OFF 이력(최근순).")
    log_list = st.session_state.get(K_CONTROL_LOG, [])
    if not log_list:
        st.caption("아직 제어 이력이 없어요.")
        return
    from control.actuators import DEVICE_LABEL_KR
    rows = []
    for log in reversed(log_list[-_MAX_LOG_ROWS:]):
        if isinstance(log, dict):
            rows.append({"날짜": log["date"], "장치": "긴급", "동작": log["action"],
                         "사유": log["reason"], "모드": log["mode"]})
        else:
            rows.append({"날짜": log.date, "장치": DEVICE_LABEL_KR.get(log.device, log.device),
                         "동작": log.action, "사유": log.reason, "모드": log.mode})
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_emergency_feed():
    alert = st.session_state.get("_last_emergency")
    if not alert:
        return
    section("🚨 긴급 경보")
    alert_box(alert["level"], f"[{alert['level']}] {alert['reason']} ({alert['date']})")
    if not alert.get("sent"):
        st.caption(f"디스코드 발송: {alert.get('msg', '-')}")


def render_forecast_row(r, date):
    from llm import expect as expect_mod

    section("🎯 오늘 예측 vs 실측")
    exp = expect_mod.expected(r, date)
    if exp is None:
        unavailable("오늘 예측", "기대값 모델(models/env_expect_reg.pkl) 없음")
    else:
        metric_row([
            ("실측 내부온도", f"{r['온도내부_평균']:.1f}℃", None),
            ("기대 내부온도", f"{exp['평균']:.1f}℃", None),
            ("잔차", f"{r['온도내부_평균'] - exp['평균']:+.1f}℃", None),
        ])

    from llm import weather as kma_weather
    try:
        current = kma_weather.get_current()
    except Exception:
        current = {"unavailable": True, "reason": "날씨 조회 중 오류"}
    if current.get("unavailable"):
        unavailable("외부 날씨(실시간)", current.get("reason", "알 수 없는 오류"))
    else:
        def _v(x, unit=""):
            return "-" if x is None else f"{x}{unit}"
        st.caption(f"실시간 외기 참고 — {_v(current.get('temp'), '℃')} · "
                   f"습도 {_v(current.get('humidity'), '%')} (재생 날짜와 다를 수 있어요)")


def render_discord_settings():
    section("📣 디스코드 알림", "긴급 경보는 항상 발송돼요. 제어 이벤트(ON/OFF) 발송은 선택이에요.")
    st.session_state[K_DISCORD_CONTROL_TOGGLE] = st.toggle(
        "장치 ON/OFF 이벤트도 디스코드로 보내기",
        value=st.session_state.get(K_DISCORD_CONTROL_TOGGLE, False))


def render():
    page_header("🌡️ 환경 관제", "설정 밴드를 벗어나면 장치가 자동으로 대응해요(시뮬) — 제어 로그·긴급 알림까지 한 화면에서.")

    vs, live, r = render_sensor_controls()
    if vs is None:
        return

    setpoints = _get_setpoints()
    states = _get_device_states()
    _run_control_step(vs, r, setpoints, states)

    render_forecast_row(r, vs.date())
    st.divider()
    render_setpoints(setpoints)
    st.divider()
    render_devices(states, setpoints, r, vs.date())
    st.divider()
    render_trend(vs, live, setpoints)
    st.divider()
    render_control_log()
    render_emergency_feed()
    st.divider()
    render_discord_settings()

    st.divider()
    st.caption("규칙 기반 자동 제어(LLM 미사용) — 히스테리시스 데드밴드로 채터링을 방지해요.")


if __name__ == "__main__":
    st.set_page_config(page_title="환경 관제", page_icon="🌡️", layout="wide")
    render()
