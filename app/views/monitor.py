"""환경 관제 페이지 — 가상 센서 재생·설정 밴드 기반 장치 자동제어(시뮬)·제어 로그·긴급 알림.

구 app/phase3_llm.py 하단부(가상 센서 ~ 조기 경보)를 분리 이관(이슈 #10 C2) 이후,
관제형 대시보드로 전면 개편(이슈 #17) — 규칙 기반 자동 제어(src/control/)와 제어 로그·
긴급 경보를 중심으로 재구성한다. vsensor 생성/시나리오 적용/tick/슬라이더 seek 로직은
문자 그대로 유지(회귀 방지) — 감싸는 것은 state.get_vsensor 뿐, 조건 분기·순서는 그대로.
오늘(실제 날짜) 기준 "오늘 운영" 탭(이슈 #23)을 추가 — KMA 외기+기대값 모델로 오늘
시간대별 제어 전/후 내부 온·습도를 보여준다(src/control/live.py, 규칙 기반). 기존
리플레이 시뮬레이션은 "🧪 시뮬레이션" 탭으로 그대로 이동(로직 변경 없음).
"""
from copy import deepcopy
from datetime import date as _date
from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from state import (K_CONTROL_ACTIVE, K_CONTROL_LAST_DATE, K_CONTROL_LOG,
                    K_CONTROL_YEAR, K_DEVICE_STATES, K_DISCORD_CONTROL_TOGGLE,
                    K_LIVE_DEVICE_STATES, K_RECENT_READINGS, K_SETPOINTS,
                    get_vsensor)
from ui import alert_box, metric_row, page_header, section, unavailable

_MAX_RECENT = 10       # emergency() 판정에 필요한 3틱보다 여유 있게 보관
_MAX_LOG_ROWS = 30      # 제어 로그 테이블 최근 N건만 표시


@st.cache_data(ttl=60)
def _cached_kma_current():
    """weather.get_current() UI 반복 방어층(이슈 #29 P2) — Streamlit rerun마다 동기 KMA
    호출이 최악 34.5s 블로킹될 수 있어 뷰 레이어에 60s 캐시를 씌운다. weather.py 내부
    TTL 캐시(10분)와 별개로, rerun 빈도가 그보다 훨씬 잦은 상황의 반복 블로킹을 막는
    용도. 콜드스타트 최초 1회 블로킹은 수용."""
    from llm import weather as kma_weather
    return kma_weather.get_current()


@st.cache_data(ttl=60)
def _cached_today_outdoor():
    """live.today_outdoor() UI 반복 방어층(이슈 #29 P2) — 위 _cached_kma_current()와 동일 이유."""
    from control import live as live_mod
    return live_mod.today_outdoor()


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

    def _ensure_scenario():
        """시나리오 선택이 실제로 바뀐 경우에만 재적용(이슈 #17 P1-1/P3).

        apply_scenario()는 "scenario" 태그 주입만 clear하므로 제어효과("control" 태그)엔
        영향 없지만, 매 rerun/틱마다 불필요하게 재호출하지 않도록 변경 시에만 실행한다."""
        if st.session_state.get(f"{scen_key}_applied") != st.session_state[scen_key]:
            apply_scenario(vs, st.session_state[scen_key])
            st.session_state[f"{scen_key}_applied"] = st.session_state[scen_key]

    _ensure_scenario()

    # 슬라이더 key는 연도별로 분리 — 연도 바뀌면 새 vs.date()로 자연 초기화됨
    date_key = f"vsensor_date_{year}"
    if date_key not in st.session_state:
        st.session_state[date_key] = vs.date()

    def _on_seek_change():
        """슬라이더 이동 → 커서 seek (버튼과 동일한 vs 인스턴스 공유)."""
        vs.seek(st.session_state[date_key])
        _ensure_scenario()

    with bcol:
        st.write("")
        if st.button("다음 날 ▶", use_container_width=True):
            vs.tick()
            _ensure_scenario()
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
    from control.setpoints import load
    if K_SETPOINTS not in st.session_state:
        st.session_state[K_SETPOINTS] = load()
    return st.session_state[K_SETPOINTS]


def _reset_control_state_for_year(year):
    """연도(작기) 변경 시 제어 상태 리셋(이슈 #17 P2-1) — 다른 작기의 실측이 emergency()의
    "3틱 연속" 판정에 섞여 들지 않도록 장치 상태·로그·긴급 dedup·최근 판독값을 초기화한다.
    설정 밴드(K_SETPOINTS)는 온실 설비 설정이라 연도와 무관 — 유지."""
    from control.actuators import default_states
    st.session_state[K_DEVICE_STATES] = default_states()
    st.session_state[K_CONTROL_LOG] = []
    st.session_state[K_CONTROL_ACTIVE] = set()
    st.session_state[K_RECENT_READINGS] = []
    st.session_state[K_CONTROL_LAST_DATE] = None
    st.session_state[K_CONTROL_YEAR] = year


def _get_device_states(year):
    from control.actuators import DEVICES
    stale_keys = (K_DEVICE_STATES in st.session_state
                  and set(st.session_state[K_DEVICE_STATES]) != set(DEVICES))
    if st.session_state.get(K_CONTROL_YEAR) != year or stale_keys:
        _reset_control_state_for_year(year)
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

    # 자기 정리(self-cleaning) dedup — active를 매 틱 "현재 만족 키셋"으로 교체해야
    # 조건 해소 후 재발 시 재발송된다(이슈 #17 P1-2, src/llm/monitor.evaluate와 동일 패턴).
    to_send, keys = controller.emergency(recent, setpoints, states, active)
    st.session_state[K_CONTROL_ACTIVE] = keys
    for alert in to_send:
        ok, msg = notify.send_discord(_emergency_embed(alert, vs.date()))
        log_list.append({"date": vs.date(), "device": "-", "action": "긴급",
                          "reason": alert["reason"], "mode": "system"})
        st.session_state["_last_emergency"] = {**alert, "date": vs.date(), "sent": ok, "msg": msg}

    st.session_state[K_CONTROL_LAST_DATE] = year_date
    return log_list


def render_setpoints(setpoints):
    from copy import copy

    from control.setpoints import save_changed

    section("⚙️ 설정 밴드", "히스테리시스 데드밴드로 장치가 경계에서 깜빡이지 않게 해요.")
    prev = copy(setpoints)
    c1, c2 = st.columns(2)
    with c1:
        setpoints.temp_low, setpoints.temp_high = st.slider(
            "온도 밴드(℃)", 0.0, 40.0, (setpoints.temp_low, setpoints.temp_high), 0.5)
    with c2:
        setpoints.hum_low, setpoints.hum_high = st.slider(
            "습도 밴드(%)", 0.0, 100.0, (setpoints.hum_low, setpoints.hum_high), 1.0)
    if setpoints != prev:
        # 동시 세션 lost-update 방지(이슈 #21 P2-2): 파일의 최신값을 베이스로
        # 이번에 실제로 바뀐 필드만 병합해 저장 — merged 결과를 세션에도 반영.
        merged = save_changed(setpoints, prev)
        setpoints.temp_low = merged.temp_low
        setpoints.temp_high = merged.temp_high
        setpoints.hum_low = merged.hum_low
        setpoints.hum_high = merged.hum_high


def _get_live_device_states():
    """오늘 운영 탭 전용 장치 상태(이슈 #25) — 시뮬용 K_DEVICE_STATES와 분리 보관.

    자정이 지나 날짜가 바뀌면 새로 초기화(전날 수동 오버라이드가 새 하루로 넘어가지 않도록).
    이슈 #27: 장치 구성이 바뀌면(예: 서버 재기동 없이 세션만 남아 옛 vent 키가 남는 경우)
    키셋이 DEVICES와 다를 때도 방어적으로 리셋한다."""
    from control.actuators import DEVICES, default_states

    today = _date.today()
    stale_keys = (K_LIVE_DEVICE_STATES in st.session_state
                  and set(st.session_state[K_LIVE_DEVICE_STATES]) != set(DEVICES))
    if (K_LIVE_DEVICE_STATES not in st.session_state
            or st.session_state.get("_live_device_states_date") != today
            or stale_keys):
        st.session_state[K_LIVE_DEVICE_STATES] = default_states()
        st.session_state["_live_device_states_date"] = today
    return st.session_state[K_LIVE_DEVICE_STATES]


def render_live_devices(states, devices_on):
    """오늘 운영 탭 장치 카드 — 자동/수동 토글·수동 ON/OFF가 오늘 운영 판정(simulate_control)에
    반영된다(이슈 #25). ON/OFF 뱃지는 현재 시각 타임라인 항목의 devices_on 기준으로 표시."""
    from control.actuators import DEVICE_LABEL_KR, DEVICES

    section("🔌 장치", "자동 모드는 설정 밴드에 따라 자동으로 켜지고 꺼져요. "
            "수동으로 전환하면 오늘 운영 판정에도 그대로 반영돼요.")
    cols = st.columns(4)
    for col, device in zip(cols, DEVICES):
        state = states[device]
        with col:
            st.markdown(f"**{DEVICE_LABEL_KR[device]}**")
            status_badge = "🟢 ON" if device in devices_on else "⚪ OFF"
            st.caption(status_badge + (" · 자동" if state.auto else " · 수동"))
            new_auto = st.toggle("자동 모드", value=state.auto, key=f"live_auto_{device}")
            if new_auto != state.auto:
                state.auto = new_auto
            if not state.auto:
                if st.button(("끄기" if state.on else "켜기"), key=f"live_manual_{device}",
                              use_container_width=True):
                    state.on = not state.on


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

    section("🎯 기대값 vs 실측")
    exp = expect_mod.expected(r, date)
    if exp is None:
        unavailable("오늘 예측", "기대값 모델(models/env_expect_reg.pkl) 없음")
    else:
        metric_row([
            ("실측 내부온도", f"{r['온도내부_평균']:.1f}℃", None),
            ("기대 내부온도", f"{exp['평균']:.1f}℃", None),
            ("잔차", f"{r['온도내부_평균'] - exp['평균']:+.1f}℃", None),
        ])

    try:
        current = _cached_kma_current()
    except Exception:
        current = {"unavailable": True, "reason": "날씨 조회 중 오류"}
    if current.get("unavailable"):
        unavailable("외부 날씨(실시간)", current.get("reason", "알 수 없는 오류"))
    else:
        def _v(x, unit=""):
            return "-" if x is None else f"{x}{unit}"
        if current.get("stale"):
            st.caption("⏳ 기상청 갱신 지연 — 직전 조회 데이터 표시 중")
        st.caption(f"실시간 외기 참고 — {_v(current.get('temp'), '℃')} · "
                   f"습도 {_v(current.get('humidity'), '%')} (재생 날짜와 다를 수 있어요)")


def render_discord_settings():
    section("📣 디스코드 알림", "긴급 경보는 항상 발송돼요. 제어 이벤트(ON/OFF) 발송은 선택이에요.")
    st.session_state[K_DISCORD_CONTROL_TOGGLE] = st.toggle(
        "장치 ON/OFF 이벤트도 디스코드로 보내기",
        value=st.session_state.get(K_DISCORD_CONTROL_TOGGLE, False))


def _live_trend_chart(rows: list, value_cols: list, low: float, high: float, now_hour: int):
    """오늘 0~24시 추이 차트(이슈 #25 사용자 추가 요청) — 라인 + 밴드 상/하한 점선 +
    "지금" 세로선 + 지금 이후(예보 구간) 음영으로 실측/예보 구간을 시각적으로 구분한다."""
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(rows)
    max_hour = float(df["hour"].max()) if not df.empty else 24.0

    forecast_band = alt.Chart(pd.DataFrame({"start": [now_hour], "end": [max_hour + 1]})).mark_rect(
        opacity=0.10, color="#888888"
    ).encode(x=alt.X("start:Q", scale=alt.Scale(domain=[0, max_hour])), x2="end:Q")

    long_df = df.melt(id_vars=["hour"], value_vars=value_cols, var_name="구분", value_name="값")
    lines = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X("hour:Q", title="시간(시)", scale=alt.Scale(domain=[0, max_hour])),
        y=alt.Y("값:Q", title=None),
        color=alt.Color("구분:N", title=None),
    )

    bounds = alt.Chart(pd.DataFrame({"y": [low, high]})).mark_rule(
        strokeDash=[2, 2], color="gray"
    ).encode(y="y:Q")

    now_rule = alt.Chart(pd.DataFrame({"hour": [now_hour]})).mark_rule(
        color="crimson", strokeDash=[4, 2], strokeWidth=2
    ).encode(x="hour:Q")

    return (forecast_band + lines + bounds + now_rule).properties(height=280)


def _split_events(items: list, now_hour: int) -> tuple:
    """timeline 이벤트류(제어 이벤트 행 또는 긴급/이상 항목)를 now_hour 기준 실행분/예정분으로
    분리한다(이슈 #31) — hour<=now_hour는 실행(done), hour>now_hour는 예정(planned).
    items는 "hour" 키를 가진 dict의 리스트."""
    done = [item for item in items if item["hour"] <= now_hour]
    planned = [item for item in items if item["hour"] > now_hour]
    return done, planned


def render_live_tab(setpoints):
    """오늘(실제 날짜) 운영 탭(이슈 #23, 레이아웃 재배치 이슈 #25) — KMA 외기+기대값
    모델로 오늘 시간대별 제어 전/후 내부 온·습도 시뮬레이션(src/control/live.py, 규칙
    기반). KMA unavailable이면 안내 후 🧪 시뮬레이션 탭으로 유도.

    장치 카드의 자동/수동 토글·수동 ON/OFF는 세션 전용 K_LIVE_DEVICE_STATES에 반영되어
    simulate_control()에 그대로 전달된다 — 단, 이는 앱 세션 한정이며 run_notify()(서버
    타이머 경로)는 세션을 모르므로 항상 default_states()(전체 자동)로 판정한다."""
    from control import live as live_mod
    from control.actuators import DEVICE_LABEL_KR

    outdoor = _cached_today_outdoor()
    if outdoor is None:
        unavailable("오늘 운영", "KMA 외기 조회 실패(키 미설정 또는 API 오류)")
        st.caption("🧪 시뮬레이션 탭에서 리플레이로 동작을 확인할 수 있어요.")
        return

    today = _date.today()
    states = _get_live_device_states()
    baseline = live_mod.indoor_baseline(outdoor, date=today)
    # simulate_control()은 전달받은 states를 시간대별로 in-place mutate한다 — 세션 원본
    # states를 그대로 넘기면 다음 리런의 "0시 시작" 상태가 직전 리런의 "23시 결과"로
    # 오염돼 auto 장치 타임라인이 리런마다 드리프트한다(P1). 시뮬용 사본을 deepcopy로
    # 분리해 세션에는 사용자가 조작한 auto/수동 값만 남긴다 — 카드 표시·토글은 원본 states.
    sim_states = deepcopy(states)
    timeline = live_mod.simulate_control(baseline, setpoints, sim_states, date=today)
    emg = live_mod.emergency_hours(timeline, setpoints)

    if not timeline:
        st.caption("오늘 시간대별 데이터가 없어요.")
        return

    now_hour = datetime.now().hour
    cur = next((t for t in timeline if t["hour"] == now_hour), timeline[-1])
    devices_on_label = ", ".join(DEVICE_LABEL_KR.get(d, d) for d in cur["devices_on"]) or "-"

    def _fmt(v, unit=""):
        return "-" if v is None else f"{v:.1f}{unit}"

    metric_row([
        ("외기 실황", _fmt(cur["out_temp"], "℃"), None),
        ("예측 내부(제어 전)", _fmt(cur["base_temp"], "℃"), None),
        ("예측 내부(제어 후)", _fmt(cur["ctrl_temp"], "℃"), None),
        ("작동 중 장치", devices_on_label, None),
    ])
    st.caption("🔮 지금 이후 값은 기상청(KMA) 예보 기반 예측이에요 — 매시 자동 갱신돼요.")

    render_setpoints(setpoints)
    render_live_devices(states, cur["devices_on"])

    section("📈 오늘 0~24시 추이", "외기·제어 전 기준선·제어 후 내부값 — 밴드를 벗어나면 장치가 자동으로 대응해요.")
    rows = [{"hour": t["hour"], "외기": t["out_temp"], "제어 전(기준선)": t["base_temp"],
             "제어 후": t["ctrl_temp"],
             "제어 전 습도(기준선)": t["base_hum"], "제어 후 습도": t["ctrl_hum"]} for t in timeline]
    temp_col, hum_col = st.columns(2)
    with temp_col:
        st.altair_chart(_live_trend_chart(
            rows, ["외기", "제어 전(기준선)", "제어 후"],
            setpoints.temp_low, setpoints.temp_high, now_hour), use_container_width=True)
        st.caption("점선=지금 · 음영=지금 이후(기상청 예보 기반 예측)")
    with hum_col:
        st.altair_chart(_live_trend_chart(
            rows, ["제어 전 습도(기준선)", "제어 후 습도"],
            setpoints.hum_low, setpoints.hum_high, now_hour), use_container_width=True)
        st.caption("점선=지금 · 음영=지금 이후(기상청 예보 기반 예측)")

    events = [{"hour": t["hour"], "시간": f"{t['hour']:02d}시",
               "장치": DEVICE_LABEL_KR.get(log.device, log.device),
               "동작": log.action, "사유": log.reason}
              for t in timeline for log in t["events"]]
    events_done, events_planned = _split_events(events, now_hour)

    section("🧾 오늘 제어 이벤트(실행)")
    if events_done:
        st.dataframe([{k: v for k, v in e.items() if k != "hour"} for e in events_done],
                      hide_index=True, use_container_width=True)
    else:
        st.caption("아직 실행된 제어가 없어요.")

    if events_planned:
        section("🔮 예정된 제어(예보 기반)", "기상청 예보 기반 예상 — 매시 갱신되며 실제 실행과 다를 수 있어요")
        st.dataframe([{k: v for k, v in e.items() if k != "hour"} for e in events_planned],
                      hide_index=True, use_container_width=True)

    if emg:
        emg_done, emg_planned = _split_events(emg, now_hour)
        section("🚨 긴급/이상")
        for item in emg_done:
            alert_box("경고", f"{item['hour']:02d}시 — {item['reason']}")
        for item in emg_planned:
            alert_box("주의", f"🔮 사전 경보(예보) {item['hour']:02d}시 — {item['reason']}")

    render_discord_settings()


def render():
    page_header("🌡️ 환경 관제", "오늘 실제 운영 상태와 시뮬레이션을 한 화면에서 확인해요.")

    setpoints = _get_setpoints()

    tab_live, tab_sim = st.tabs(["🟢 오늘 운영", "🧪 시뮬레이션"])

    with tab_live:
        render_live_tab(setpoints)

    with tab_sim:
        vs, live, r = render_sensor_controls()
        if vs is not None:
            states = _get_device_states(vs.year)
            _run_control_step(vs, r, setpoints, states)

            render_forecast_row(r, vs.date())
            st.divider()
            render_trend(vs, live, setpoints)
            st.divider()
            render_control_log()
            render_emergency_feed()

    st.divider()
    st.caption("규칙 기반 자동 제어(LLM 미사용) — 히스테리시스 데드밴드로 채터링을 방지해요.")


if __name__ == "__main__":
    st.set_page_config(page_title="환경 관제", page_icon="🌡️", layout="wide")
    render()
