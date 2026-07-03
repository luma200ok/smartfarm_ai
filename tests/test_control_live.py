"""src/control/live.py — 오늘 운영 모드(이슈 #23) 테스트. KMA·expect는 mock."""
import json

import pytest

from control import live
from control.actuators import default_states
from control.setpoints import Setpoints


def _sp():
    return Setpoints()  # 온도 20~25℃, 습도 60~85%


def _forecast(hours_temp: dict, date_str="20260703"):
    """hour(int) -> temp 매핑으로 get_forecast_3d() 형태 mock 응답 구성."""
    hourly = [{"date": date_str, "time": f"{h:02d}00", "temp": t, "humidity": 70.0}
              for h, t in hours_temp.items()]
    return {"unavailable": False, "hourly": hourly, "daily": []}


# ── today_outdoor ────────────────────────────────────────────────────────
def test_today_outdoor_none_when_kma_unavailable(monkeypatch):
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: {"unavailable": True, "reason": "키 없음"})
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    assert live.today_outdoor() is None


def test_today_outdoor_replaces_current_hour(monkeypatch):
    from datetime import date, datetime
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 20.0 + h * 0.1 for h in range(24)}))
    monkeypatch.setattr(weather, "get_current",
                         lambda: {"unavailable": False, "temp": 99.0, "humidity": 55.0})
    now = datetime(2026, 7, 3, 14, 30)
    out = live.today_outdoor(today=date(2026, 7, 3), now=now)
    assert out is not None
    item14 = next(i for i in out if i["hour"] == 14)
    assert item14["temp"] == 99.0
    assert item14["humidity"] == 55.0
    assert len(out) == 24


# ── indoor_baseline ──────────────────────────────────────────────────────
def _patch_expect_model(monkeypatch, slope=0.8, intercept=2.0):
    """expect.load_model()이 predict()에서 온도외부_평균*slope+intercept를 돌려주는 더미 모델."""
    from llm import expect as expect_mod

    class _DummyModel:
        def predict(self, X):
            import numpy as np
            return np.array([X[0][0] * slope + intercept])

    payload = {
        "features": ["온도외부_평균", "일사량_평균", "doy_sin", "doy_cos"],
        "models": {"평균": _DummyModel(), "최저": _DummyModel()},
        "resid_sigma": {"평균": 1.0, "최저": 1.0},
        "doy_solar_climatology": {},
    }
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: payload)


def test_indoor_baseline_uses_expect_model(monkeypatch):
    from datetime import date
    _patch_expect_model(monkeypatch)
    outdoor = [{"hour": 12, "temp": 30.0, "humidity": 50.0}]
    baseline = live.indoor_baseline(outdoor, date=date(2026, 7, 3))
    assert baseline[0]["base_temp"] == pytest.approx(30.0 * 0.8 + 2.0)
    assert baseline[0]["base_hum"] == pytest.approx(60.0)  # 50 + INDOOR_HUMIDITY_OFFSET(10)


def test_indoor_baseline_uses_model_humidity_when_available(monkeypatch):
    """이슈 #37 — 모델 payload에 "습도" 타깃이 있으면 base_hum은 predict()의 습도값을
    그대로 쓴다(기존 외기+OFFSET 폴백 대신)."""
    from datetime import date
    from llm import expect as expect_mod

    class _DummyHumModel:
        def predict(self, X):
            import numpy as np
            return np.array([42.0])

    payload = {
        "features": ["온도외부_평균", "일사량_평균", "doy_sin", "doy_cos"],
        "models": {"평균": _DummyHumModel(), "최저": _DummyHumModel(), "습도": _DummyHumModel()},
        "resid_sigma": {"평균": 1.0, "최저": 1.0, "습도": 1.0},
        "doy_solar_climatology": {},
    }
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: payload)
    outdoor = [{"hour": 12, "temp": 30.0, "humidity": 50.0}]
    baseline = live.indoor_baseline(outdoor, date=date(2026, 7, 3))
    # 폴백(50+10=60)이 아니라 모델 예측값(42.0)이 그대로 쓰여야 함
    assert baseline[0]["base_hum"] == pytest.approx(42.0)


def test_indoor_baseline_falls_back_to_outdoor_temp_when_no_model(monkeypatch):
    from datetime import date
    from llm import expect as expect_mod
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: None)
    outdoor = [{"hour": 12, "temp": 33.0, "humidity": 40.0}]
    baseline = live.indoor_baseline(outdoor, date=date(2026, 7, 3))
    assert baseline[0]["base_temp"] == 33.0


# ── simulate_control ─────────────────────────────────────────────────────
def _baseline(hours):
    """hour -> (base_temp, base_hum) dict → baseline 리스트."""
    return [{"hour": h, "out_temp": t, "out_hum": hum, "base_temp": t, "base_hum": hum}
            for h, (t, hum) in hours.items()]


def test_simulate_control_turns_on_device_when_over_band_and_moves_ctrl_toward_band():
    # 매 시간 30℃(밴드 상한 25℃ 초과) 유지 — cooling_fan/vent가 계속 ON, ctrl_temp가
    # base_temp보다 낮아져야 함(냉방 효과로 밴드 방향 조정).
    baseline = _baseline({h: (30.0, 70.0) for h in range(5)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    # 첫 시간엔 ctrl_temp==base_temp(기준선 그대로 시작)이고, 온도가 밴드 상한을 넘으므로
    # cooling_fan·vent가 즉시 ON 판정돼야 한다.
    assert timeline[0]["ctrl_temp"] == pytest.approx(30.0)
    assert "cooling_fan" in timeline[0]["devices_on"]
    later = timeline[-1]
    assert later["ctrl_temp"] < later["base_temp"]


def test_simulate_control_devices_off_within_band():
    baseline = _baseline({h: (22.0, 70.0) for h in range(3)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    for item in timeline:
        assert item["devices_on"] == []
        assert item["events"] == []


def test_simulate_control_emergency_when_full_blast_cannot_recover():
    # 밴드 상한을 훨씬 초과하는 외기·기준선이 지속돼 냉방+환기 풀가동으로도 못 잡는 상황을 구성.
    baseline = _baseline({h: (45.0, 70.0) for h in range(6)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    emg = live.emergency_hours(timeline, _sp())
    assert len(emg) > 0
    assert all("고온 지속" in e["reason"] for e in emg)


def test_simulate_control_ctrl_temp_clamped_within_band(monkeypatch):
    """물리 클램프(이슈 #27) — 극단적으로 낮은 base_temp(난방 지속)에도 ctrl_temp가
    base_temp - CTRL_TEMP_BAND보다 아래로 발산하지 않아야 한다."""
    baseline = _baseline({h: (-30.0, 70.0) for h in range(4)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    for item in timeline[1:]:
        assert item["ctrl_temp"] >= -30.0 - live.CTRL_TEMP_BAND - 1e-9


def test_simulate_control_ctrl_hum_clamped_0_100():
    """제습기 지속 가동에도 ctrl_hum이 0% 아래로, 가습기 지속에도 100% 위로 발산하지 않는다."""
    baseline_low = _baseline({h: (22.0, 0.5) for h in range(6)})
    timeline_low = live.simulate_control(baseline_low, _sp(), default_states(), date="2026-07-03")
    assert all(item["ctrl_hum"] >= 0.0 for item in timeline_low)

    baseline_high = _baseline({h: (22.0, 99.5) for h in range(6)})
    timeline_high = live.simulate_control(baseline_high, _sp(), default_states(), date="2026-07-03")
    assert all(item["ctrl_hum"] <= 100.0 for item in timeline_high)


def test_simulate_control_hum_converges_to_band_dramatic_effect():
    """드라마틱 효과 검증(이슈 #27) — 고습 프로파일(외기 습도가 시간에 따라 낮아지는 하루
    추이)에서 제습기 ON → ctrl_hum이 몇 시간 내 밴드(hum_high=85) 아래로 수렴한다."""
    hums = {0: 95.0, 1: 90.0, 2: 85.0, 3: 80.0, 4: 75.0, 5: 70.0}
    baseline = _baseline({h: (22.0, hums[h]) for h in range(6)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    assert "dehumidifier" in timeline[0]["devices_on"]
    # 제습 효과(-8.0%p/h)로 몇 시간 내 밴드 상한(85%) 아래로 수렴해야 한다.
    assert any(item["ctrl_hum"] < 85.0 for item in timeline[1:])


# ── 습도 P-제어(이슈 #33) ─────────────────────────────────────────────────
def test_simulate_control_hum_converges_to_mid_no_chattering():
    """고습 시작 → 제습기 ON → ctrl_hum이 밴드 중앙(72.5%) 부근으로 수렴하고 진동 없음."""
    baseline = _baseline({h: (22.0, 90.0) for h in range(30)})  # 외기 습도 고정(안정적 수렴 확인)
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    on_flags = ["dehumidifier" in t["devices_on"] for t in timeline]
    transitions = sum(1 for i in range(1, len(on_flags)) if on_flags[i] != on_flags[i - 1])
    assert transitions <= 2  # ON → (수렴 후) OFF, 그 이상 왔다갔다 없음(진동 방지)

    last = timeline[-1]
    hum_mid = (_sp().hum_low + _sp().hum_high) / 2
    assert abs(last["ctrl_hum"] - hum_mid) <= _sp().hum_deadband + 1e-6


def test_simulate_control_hum_delta_shrinks_near_mid():
    """중앙 근접 시 델타(비례)가 감소 — 초기 큰 오차 스텝보다 중앙 근접 스텝의 변화폭이 작다."""
    baseline = _baseline({h: (22.0, 90.0) for h in range(10)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    early_step = abs(timeline[1]["ctrl_hum"] - timeline[0]["ctrl_hum"])
    late_step = abs(timeline[-1]["ctrl_hum"] - timeline[-2]["ctrl_hum"])
    assert late_step < early_step


def test_simulate_control_hum_delta_capped_at_max():
    """오차가 클 때(캡 도달 수준) 한 스텝 변화가 HUM_P_MAX_DELTA를 넘지 않는다."""
    baseline = _baseline({h: (22.0, 99.0) for h in range(3)})  # 오차 큼(밴드 상한보다 훨씬 위)
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    step = timeline[0]["ctrl_hum"] - timeline[1]["ctrl_hum"] if timeline[1]["ctrl_hum"] is not None else 0
    assert step <= live.HUM_P_MAX_DELTA + 1e-9


def test_simulate_control_hum_converges_symmetric_humidifier():
    """저습 대칭 — 가습기 ON → ctrl_hum이 중앙(72.5%)으로 수렴."""
    baseline = _baseline({h: (22.0, 40.0) for h in range(30)})  # 저습 고정
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    on_flags = ["humidifier" in t["devices_on"] for t in timeline]
    transitions = sum(1 for i in range(1, len(on_flags)) if on_flags[i] != on_flags[i - 1])
    assert transitions <= 2

    last = timeline[-1]
    hum_mid = (_sp().hum_low + _sp().hum_high) / 2
    assert abs(last["ctrl_hum"] - hum_mid) <= _sp().hum_deadband + 1e-6


def test_simulate_control_emergency_hum_still_detected_with_pcontrol():
    """P-제어 도입으로 긴급(풀가동에도 못 잡음) 오판이 없는지 확인 — 긴급 판정 시점엔
    밴드 밖(오차 큼)이라 델타가 캡(±8)에 있어 emergency_hours() 의미가 그대로 유지된다."""
    baseline = _baseline({h: (22.0, 99.0) for h in range(6)})  # 밴드 상한(85%)을 훨씬 초과
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    emg = live.emergency_hours(timeline, _sp())
    assert len(emg) > 0
    assert all("고습 지속" in e["reason"] for e in emg)


def test_dehumidifier_and_humidifier_never_both_on_live():
    from control import controller
    states = default_states()
    controller.decide({"온도내부_평균": 22.0, "습도내부_평균": 95.0}, _sp(), states, date="d1")
    assert not (states["dehumidifier"].on and states["humidifier"].on)
    controller.decide({"온도내부_평균": 22.0, "습도내부_평균": 30.0}, _sp(), states, date="d2")
    assert not (states["dehumidifier"].on and states["humidifier"].on)


def test_state_file_vent_key_backward_compat(monkeypatch, _isolated_state, _isolated_setpoints):
    """상태 파일에 옛 vent 키가 남아 있어도 예외 없이 신규 장치 키(dehumidifier)로
    재구성돼야 한다(이슈 #27)."""
    from datetime import date
    from llm import weather

    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 22.0 for h in range(24)}, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    stale = {"date": "2026-07-03", "fail_count": 0,
             "devices": {"vent": True, "heater": False, "cooling_fan": False, "humidifier": False},
             "emergency_hours": []}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(stale), encoding="utf-8")

    sent = []
    _patch_notify(monkeypatch, sent)

    # 예외 없이 실행돼야 한다 — 옛 vent 키는 무시되고 신규 DEVICES 키셋으로 재구성.
    live.run_notify(dry_run=False, today=date(2026, 7, 3))
    state = json.loads(_isolated_state.read_text())
    assert set(state["devices"]) == {"dehumidifier", "humidifier", "cooling_fan", "heater"}


def test_simulate_control_no_chattering_with_variable_baseline():
    """리뷰 P2-1 픽스 검증 — 가변 외기(밴드 경계 부근에서 ±1~1.5℃ 흔들림)에서도 제어
    관성(누적) 방식 + 히스테리시스로 장치 ON/OFF 전환 횟수가 임계(3회) 이하여야 한다."""
    base = 25.3  # 밴드 상한(25.0) 바로 위 — 경계 부근
    swing = [1.5, -1.2, 1.3, -1.0, 1.4, -1.1, 1.2, -1.3, 1.1, -1.4]
    hours_temp = {}
    v = base
    for h, s in enumerate(swing):
        v = base + s
        hours_temp[h] = v
    baseline = _baseline({h: (t, 70.0) for h, t in hours_temp.items()})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    on_flags = ["cooling_fan" in t["devices_on"] for t in timeline]
    transitions = sum(1 for i in range(1, len(on_flags)) if on_flags[i] != on_flags[i - 1])
    assert transitions <= 3


def test_simulate_control_no_single_step_penetration_temp():
    """리뷰 P2-1 픽스 검증 — 고온 시작(냉방 ON) 후 외기가 급락해도 한 스텝에 temp_low
    아래로 관통해 heater가 즉시 켜지는 교대 진동이 없어야 한다(관통 방지 클램프)."""
    # 45℃(냉방 ON 유발) → 다음 시간 -10℃로 급락(관통 시도).
    baseline = _baseline({0: (45.0, 70.0), 1: (-10.0, 70.0), 2: (-10.0, 70.0)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    assert "cooling_fan" in timeline[0]["devices_on"]
    # 1시간 뒤(관통 방지 클램프 적용된 ctrl_temp) 값 자체가 temp_low+deadband 아래로
    # 내려가지 않아야 하고, heater가 같은 스텝에서 곧바로 ON 되지 않아야 한다.
    sp = _sp()
    assert timeline[1]["ctrl_temp"] >= sp.temp_low + sp.temp_deadband - 1e-9
    assert "heater" not in timeline[1]["devices_on"]


def test_simulate_control_deterministic_when_states_deepcopied_between_calls():
    """오늘 운영 탭(app/views/monitor.py) P1 회귀 방지 — simulate_control()은 전달받은
    states를 in-place mutate하므로, 세션 원본을 그대로 재사용하면 두 번째 호출의 "0시
    시작" 상태가 첫 번째 호출의 "마지막 시간 결과"로 오염돼 타임라인이 리런마다 드리프트
    한다. monitor.py는 매 렌더 `deepcopy(states)`로 시뮬용 사본을 분리해 세션 states는
    항상 default_states() 그대로 유지한다 — 이 테스트는 그 패턴으로 동일 입력 2회 호출 시
    타임라인이 완전히 동일함을 검증한다."""
    from copy import deepcopy

    baseline = _baseline({h: (30.0, 70.0) for h in range(6)})  # 밴드 상한 초과 지속 — 장치 ON 유발
    session_states = default_states()

    sim_states_1 = deepcopy(session_states)
    timeline_1 = live.simulate_control(baseline, _sp(), sim_states_1, date="2026-07-03")

    # 세션 원본(session_states)은 첫 호출로 오염되지 않아야 한다.
    assert all(not s.on for s in session_states.values())

    sim_states_2 = deepcopy(session_states)
    timeline_2 = live.simulate_control(baseline, _sp(), sim_states_2, date="2026-07-03")

    def _strip(timeline):
        return [{k: v for k, v in item.items() if k != "events"} for item in timeline]

    assert _strip(timeline_1) == _strip(timeline_2)


def test_simulate_control_without_deepcopy_drifts_across_calls():
    """대조군 — deepcopy 없이 같은 states 객체를 재사용하면(수정 전 버그) 직전 호출이 장치를
    ON으로 남긴 채 끝난 뒤, 그 상태를 이어받아 재실행하면 히스테리시스 데드밴드 때문에
    "0시부터 밴드 정상 범위"인 새 타임라인의 첫 시간조차 잘못 ON으로 판정된다."""
    # 이슈 #27(제어 관성 방식) — 냉방 효과(-2.0℃/h)가 누적되므로 30℃로는 6시간 안에 밴드로
    # 수렴해버려 fan이 자연히 OFF된다. 시종일관 ON 상태를 유지시키려면 충분히 높은 기준선(45℃)
    # 이 필요하다(6시간 동안 2℃/h씩 내려가도 35℃로 여전히 밴드 상한 초과).
    hot_baseline = _baseline({h: (45.0, 70.0) for h in range(6)})   # 밴드 상한 초과 지속 — fan ON 유발
    normal_baseline = _baseline({h: (24.7, 70.0) for h in range(3)})  # 데드밴드(24.5) 안쪽, 밴드 자체는 정상

    states_fresh = default_states()
    fresh_timeline = live.simulate_control(normal_baseline, _sp(), states_fresh, date="2026-07-03")
    assert fresh_timeline[0]["devices_on"] == []  # 새 states로 시작하면 정상 범위라 OFF

    states_polluted = default_states()
    live.simulate_control(hot_baseline, _sp(), states_polluted, date="2026-07-03")
    assert states_polluted["cooling_fan"].on  # 직전 호출로 오염 — ON 상태로 남음

    polluted_timeline = live.simulate_control(normal_baseline, _sp(), states_polluted, date="2026-07-03")
    # 같은 normal_baseline인데 오염된 states로 재실행하면 첫 시간 devices_on이 fresh와 달라진다
    # (데드밴드 안쪽이라 was_high=True 이력이 남은 fan이 계속 ON으로 판정됨) — 회귀 재현.
    assert polluted_timeline[0]["devices_on"] != fresh_timeline[0]["devices_on"]


# ── run_notify — 상태 파일 dedup ─────────────────────────────────────────
@pytest.fixture
def _isolated_state(tmp_path, monkeypatch):
    state_path = tmp_path / "control_live_state.json"
    monkeypatch.setattr(live, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def _isolated_setpoints(tmp_path, monkeypatch):
    from control import setpoints as setpoints_mod
    sp_path = tmp_path / "control_setpoints.json"
    monkeypatch.setattr(setpoints_mod, "SETPOINTS_PATH", sp_path)
    return sp_path


def _patch_notify(monkeypatch, sent: list):
    from llm import notify
    def _fake_send(embed):
        sent.append(embed)
        return True, "ok"
    monkeypatch.setattr(notify, "send_discord", _fake_send)


def test_run_notify_sends_on_first_run_and_dedups_second_run(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """자정 연속성 폴백(이슈 #35, run_notify는 fallback_clamp=True) 하에서 0시 ctrl은
    밴드 안에서 시작하므로, 장치 전환을 보려면 baseline이 시간에 따라 밴드 밖으로
    드리프트해야 한다(과거처럼 평평한 프로파일은 클램프된 채 영원히 밴드 안에 머문다) —
    ramp 프로파일 + now=12시로 확정적으로 재현."""
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}  # 20℃(0시) → 43℃(23시), 밴드 상한 25 돌파
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    now = datetime(2026, 7, 3, 12, 0)
    n1 = live.run_notify(dry_run=False, today=today, now=now)
    assert n1 >= 1  # 첫 실행 — 드리프트로 밴드 초과, cooling_fan 등 장치 전환 발송

    n2 = live.run_notify(dry_run=False, today=today, now=now)
    assert n2 == 0  # 같은 상태 재실행 — 전환 없음, 발송 0건


def test_run_notify_uses_current_hour_not_timeline_end_for_transitions(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """P1-2 회귀 방지 — 현재 시각(now.hour)에는 밴드 내(장치 OFF)인데 미래 시간대만
    밴드 초과(장치 ON)인 프로파일. 픽스 전엔 timeline 마지막(미래) 상태를 "지금 전환"으로
    오인해 cooling_fan ON 알림을 보냈지만, 픽스 후엔 현재 시각 기준 OFF라 발송 0건이어야 한다."""
    from datetime import date, datetime
    from llm import weather

    # 09시(현재로 고정)는 밴드 내(22℃), 10시 이후만 밴드 초과(30℃) — 미래에만 장치가 켜짐.
    hours_temp = {h: (22.0 if h <= 9 else 30.0) for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(hours_temp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    now = datetime(2026, 7, 3, 9, 0)   # 09시 = 아직 밴드 내(장치 OFF)여야 하는 "지금"
    live.run_notify(dry_run=False, today=today, now=now)
    # 미래(10시 이후) 밴드 초과로 인한 긴급 알림은 별개로 발송될 수 있지만, "장치 전환"
    # 알림(🎛)은 지금(09시) 기준으로는 없어야 한다 — 픽스 전엔 timeline 마지막(미래, 30℃
    # 지속)의 cooling_fan ON을 "지금 전환"으로 오인해 이 임베드가 섞여 들어갔다.
    device_transition_embeds = [e for e in sent if e["title"].startswith("🎛")]
    assert device_transition_embeds == []


def test_run_notify_resets_on_new_date(monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    live.run_notify(dry_run=False, today=date(2026, 7, 3), now=datetime(2026, 7, 3, 12, 0))
    state = json.loads(_isolated_state.read_text())
    assert state["date"] == "2026-07-03"

    # 다음날 같은 forecast(date_str 다름으로 today_outdoor가 None 반환하지 않도록 갱신)
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260704"))
    n = live.run_notify(dry_run=False, today=date(2026, 7, 4), now=datetime(2026, 7, 4, 12, 0))
    assert n >= 1  # 날짜가 바뀌어 상태 리셋 → 다시 전환으로 인식돼 발송
    state2 = json.loads(_isolated_state.read_text())
    assert state2["date"] == "2026-07-04"


def test_run_notify_dry_run_does_not_write_state_or_call_discord(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    from llm import notify
    called = []
    monkeypatch.setattr(notify, "send_discord", lambda embed: called.append(embed) or (True, "ok"))

    n = live.run_notify(dry_run=True, today=date(2026, 7, 3), now=datetime(2026, 7, 3, 12, 0))
    assert n >= 1
    assert called == []            # 실제 discord 호출 없음
    assert not _isolated_state.exists()  # 상태 파일도 쓰지 않음


def test_run_notify_kma_unavailable_graceful(monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: {"unavailable": True, "reason": "키 없음"})
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})

    sent = []
    _patch_notify(monkeypatch, sent)

    n1 = live.run_notify(dry_run=False, today=date(2026, 7, 3))
    assert n1 == 0     # 1회차 실패는 아직 이상 알림 안 보냄(연속 2회 임계치)
    n2 = live.run_notify(dry_run=False, today=date(2026, 7, 3))
    assert n2 == 1     # 2회 연속 실패 → 이상 알림 1건


# ── 자정 연속성(이슈 #35) — initial_ctrl 시드 / 폴백 클램프 ───────────────
def test_simulate_control_initial_ctrl_seeds_hour_zero_instead_of_baseline():
    """initial_ctrl의 date가 정확히 어제면 0시 ctrl 시작값이 기준선이 아니라 시드에서
    출발한다 — 밴드 밖(30.0) 기준선이어도 시드(21.0)를 그대로 이어받아야 함."""
    from datetime import date as _dt
    baseline = _baseline({h: (30.0, 70.0) for h in range(3)})  # 밴드 상한(25) 초과 기준선
    states = default_states()
    initial_ctrl = {"date": "2026-07-02", "hour": 23, "temp": 21.0, "hum": 65.0}  # 어제
    timeline = live.simulate_control(baseline, _sp(), states, date=_dt(2026, 7, 3),
                                      initial_ctrl=initial_ctrl)
    assert timeline[0]["ctrl_temp"] == pytest.approx(21.0)
    assert timeline[0]["ctrl_hum"] == pytest.approx(65.0)


def test_simulate_control_initial_ctrl_today_date_is_ignored():
    """last_ctrl의 date가 "오늘"(어제가 아님)이면 무효 — 같은 날 재호출 시 재시딩 방지
    (run_notify 멱등성 회귀 방지)."""
    from datetime import date as _dt
    baseline = _baseline({h: (30.0, 70.0) for h in range(3)})
    states = default_states()
    today_seed = {"date": "2026-07-03", "hour": 12, "temp": 21.0, "hum": 65.0}
    timeline = live.simulate_control(baseline, _sp(), states, date=_dt(2026, 7, 3),
                                      initial_ctrl=today_seed)
    assert timeline[0]["ctrl_temp"] == pytest.approx(30.0)  # 기준선 그대로(시드 무시)


def test_simulate_control_initial_ctrl_stale_date_is_ignored():
    """last_ctrl의 date가 그제(2일 전) 이상이면 무효 — initial_ctrl 미사용(기존 거동)."""
    from datetime import date as _dt
    baseline = _baseline({h: (30.0, 70.0) for h in range(3)})
    states = default_states()
    stale = {"date": "2026-07-01", "hour": 23, "temp": 21.0, "hum": 65.0}
    timeline = live.simulate_control(baseline, _sp(), states, date=_dt(2026, 7, 3),
                                      initial_ctrl=stale)
    assert timeline[0]["ctrl_temp"] == pytest.approx(30.0)  # 기준선 그대로(시드 무시)


def test_simulate_control_fallback_clamp_starts_within_band():
    """initial_ctrl 없음 + fallback_clamp=True → 기준선이 밴드 밖(90%)이어도 0시 ctrl은
    밴드 안(hum_low+deadband ~ hum_high-deadband)에서 시작한다."""
    sp = _sp()
    baseline = _baseline({h: (30.0, 90.0) for h in range(3)})  # 온·습도 모두 밴드 초과
    states = default_states()
    timeline = live.simulate_control(baseline, sp, states, date="2026-07-03",
                                      initial_ctrl=None, fallback_clamp=True)
    assert sp.temp_low + sp.temp_deadband - 1e-9 <= timeline[0]["ctrl_temp"] <= sp.temp_high - sp.temp_deadband + 1e-9
    assert sp.hum_low + sp.hum_deadband - 1e-9 <= timeline[0]["ctrl_hum"] <= sp.hum_high - sp.hum_deadband + 1e-9


def test_simulate_control_fallback_clamp_default_off_keeps_legacy_behavior():
    """fallback_clamp 기본값(False)에서는 기존 거동(기준선 그대로 시작) 유지 — 무회귀."""
    baseline = _baseline({h: (30.0, 90.0) for h in range(3)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    assert timeline[0]["ctrl_temp"] == pytest.approx(30.0)
    assert timeline[0]["ctrl_hum"] == pytest.approx(90.0)


def test_run_notify_persists_last_ctrl_and_preserves_dedup_fields(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """run_notify가 상태 파일에 last_ctrl을 기록하고, 기존 dedup 필드(devices·
    emergency_hours)도 그대로 보존한다."""
    from datetime import date
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 30.0 for h in range(24)}, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    live.run_notify(dry_run=False, today=today)

    state = json.loads(_isolated_state.read_text(encoding="utf-8"))
    assert "last_ctrl" in state
    assert state["last_ctrl"]["date"] == "2026-07-03"
    assert state["last_ctrl"]["temp"] is not None
    assert "devices" in state and "emergency_hours" in state  # 기존 dedup 필드 보존
