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


# ── 온도 P-제어(이슈 #45, 습도와 대칭) ─────────────────────────────────────
def test_temp_pcontrol_delta_negative_when_cooling_above_mid():
    """냉방 ON + ctrl_temp가 중앙보다 높으면 델타는 음수(중앙으로 끌어내림)."""
    delta = live._temp_pcontrol_delta(["cooling_fan"], 24.0, _sp())
    assert delta < 0.0


def test_temp_pcontrol_delta_positive_when_heater_below_mid():
    """히터 ON + ctrl_temp가 중앙보다 낮으면 델타는 양수(중앙으로 끌어올림)."""
    delta = live._temp_pcontrol_delta(["heater"], 21.0, _sp())
    assert delta > 0.0


def test_temp_pcontrol_delta_zero_when_both_devices_off():
    """냉방·히터 둘 다 OFF면 장치 효과가 없으므로 델타 0.0."""
    assert live._temp_pcontrol_delta([], 30.0, _sp()) == 0.0
    assert live._temp_pcontrol_delta([], -10.0, _sp()) == 0.0


def test_temp_pcontrol_delta_capped_at_max():
    """오차가 클 때(캡 도달 수준) 델타가 TEMP_P_MAX_DELTA를 넘지 않는다(양방향)."""
    delta_cool = live._temp_pcontrol_delta(["cooling_fan"], 99.0, _sp())
    assert delta_cool == pytest.approx(-live.TEMP_P_MAX_DELTA)
    delta_heat = live._temp_pcontrol_delta(["heater"], -99.0, _sp())
    assert delta_heat == pytest.approx(live.TEMP_P_MAX_DELTA)


def test_simulate_control_temp_converges_to_mid_no_chattering():
    """고온 시작 → 냉방 ON → ctrl_temp가 밴드 중앙(22.5℃) 부근으로 수렴하고 진동 없음."""
    baseline = _baseline({h: (30.0, 70.0) for h in range(30)})  # 외기 온도 고정(안정적 수렴 확인)
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    on_flags = ["cooling_fan" in t["devices_on"] for t in timeline]
    transitions = sum(1 for i in range(1, len(on_flags)) if on_flags[i] != on_flags[i - 1])
    assert transitions <= 2  # ON → (수렴 후) OFF, 그 이상 왔다갔다 없음(진동 방지)

    last = timeline[-1]
    temp_mid = (_sp().temp_low + _sp().temp_high) / 2
    assert abs(last["ctrl_temp"] - temp_mid) <= _sp().temp_deadband + 1e-6


def test_simulate_control_temp_converges_symmetric_heater():
    """저온 대칭 — 히터 ON → ctrl_temp가 중앙(22.5℃)으로 수렴."""
    baseline = _baseline({h: (15.0, 70.0) for h in range(30)})  # 저온 고정
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    on_flags = ["heater" in t["devices_on"] for t in timeline]
    transitions = sum(1 for i in range(1, len(on_flags)) if on_flags[i] != on_flags[i - 1])
    assert transitions <= 2

    last = timeline[-1]
    temp_mid = (_sp().temp_low + _sp().temp_high) / 2
    assert abs(last["ctrl_temp"] - temp_mid) <= _sp().temp_deadband + 1e-6


# ── 중앙 유지형 재설계(이슈 #51) — 계절형 외란 지속 시나리오 ────────────────
# 재설계 전엔 외란이 한쪽으로 지속되면(예: 여름철 고습·고온) ctrl 값이 밴드 상단
# (구 기본값 60~85% 기준 74.5~85%)에만 갇혀 "중앙 위에서만 놀았다". 겨울철은 대칭으로
# 하한 정체. 아래는 이 문제가 해소돼 실제로 중앙 부근에 수렴함을 모델 없이(합성 baseline)
# 직접 검증한다.
def test_simulate_control_summer_disturbance_converges_near_mid_not_stuck_at_band_top():
    """여름형 — 외기 유입으로 base_temp·base_hum이 지속적으로 밴드 상단을 넘는 고온다습
    (27.5℃·92%)이 계속돼도, 냉방·제습 효과로 ctrl 값이 밴드 상단에 갇히지 않고 중앙
    부근(계절 특성상 중앙보다 약간 위)으로 수렴해야 한다."""
    baseline = _baseline({h: (27.5, 92.0) for h in range(48)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    sp = _sp()
    temp_mid = (sp.temp_low + sp.temp_high) / 2
    hum_mid = (sp.hum_low + sp.hum_high) / 2
    last = timeline[-1]

    # 중앙 근접 수렴 — 데드밴드 폭 이내(구버전이라면 밴드 상단에 갇혀 훨씬 크게 벗어남).
    assert abs(last["ctrl_temp"] - temp_mid) <= sp.temp_deadband + 1e-6
    assert abs(last["ctrl_hum"] - hum_mid) <= sp.hum_deadband + 1e-6
    # 계절 lean 방향 — 외란이 고온다습 쪽이므로 중앙보다 살짝 위에서 수렴.
    assert last["ctrl_temp"] >= temp_mid
    assert last["ctrl_hum"] >= hum_mid
    # 옛 정체 구간(밴드 상단, 예: 습도 74.5~85%)에 갇히지 않음 — 중앙과의 거리가
    # 밴드 반폭(high-mid)의 절반보다 훨씬 작아야 한다.
    assert abs(last["ctrl_hum"] - hum_mid) < (sp.hum_high - hum_mid) / 2
    assert abs(last["ctrl_temp"] - temp_mid) < (sp.temp_high - temp_mid) / 2

    # 채터링 없음 — 장치 상태가 수렴 후 매 시간 토글되지 않음.
    temp_on = ["cooling_fan" in t["devices_on"] for t in timeline]
    hum_on = ["dehumidifier" in t["devices_on"] for t in timeline]
    temp_transitions = sum(1 for i in range(1, len(temp_on)) if temp_on[i] != temp_on[i - 1])
    hum_transitions = sum(1 for i in range(1, len(hum_on)) if hum_on[i] != hum_on[i - 1])
    assert temp_transitions <= 2
    assert hum_transitions <= 2


def test_simulate_control_winter_disturbance_converges_near_mid_not_stuck_at_band_bottom():
    """겨울형 — base_temp·base_hum이 지속적으로 낮은 저온저습(15.0℃·40%)이 계속돼도,
    난방·가습 효과로 ctrl 값이 밴드 하단에 갇히지 않고 중앙 부근(계절 특성상 중앙보다
    약간 아래)으로 수렴해야 한다(여름형과 대칭)."""
    baseline = _baseline({h: (15.0, 40.0) for h in range(48)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")

    sp = _sp()
    temp_mid = (sp.temp_low + sp.temp_high) / 2
    hum_mid = (sp.hum_low + sp.hum_high) / 2
    last = timeline[-1]

    assert abs(last["ctrl_temp"] - temp_mid) <= sp.temp_deadband + 1e-6
    assert abs(last["ctrl_hum"] - hum_mid) <= sp.hum_deadband + 1e-6
    # 계절 lean 방향 — 외란이 저온저습 쪽이므로 중앙보다 살짝 아래에서 수렴.
    assert last["ctrl_temp"] <= temp_mid
    assert last["ctrl_hum"] <= hum_mid
    # 옛 정체 구간(밴드 하단)에 갇히지 않음.
    assert abs(last["ctrl_hum"] - hum_mid) < (hum_mid - sp.hum_low) / 2
    assert abs(last["ctrl_temp"] - temp_mid) < (temp_mid - sp.temp_low) / 2

    temp_on = ["heater" in t["devices_on"] for t in timeline]
    hum_on = ["humidifier" in t["devices_on"] for t in timeline]
    temp_transitions = sum(1 for i in range(1, len(temp_on)) if temp_on[i] != temp_on[i - 1])
    hum_transitions = sum(1 for i in range(1, len(hum_on)) if hum_on[i] != hum_on[i - 1])
    assert temp_transitions <= 2
    assert hum_transitions <= 2


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
    """리뷰 P2-1 픽스 검증 — 가변 외기(진입 문턱 부근에서 흔들림)에서도 제어 관성(누적)
    방식 + 히스테리시스로 장치 ON/OFF 전환 횟수가 임계(3회) 이하여야 한다.

    이슈 #51(중앙 유지형 재설계)로 ON 진입 문턱이 밴드 경계(25.0℃)에서 중앙+데드밴드
    (22.5+0.5=23.0℃)로 훨씬 좁아졌다 — 그만큼 흔들림 진폭도 문턱 부근 스케일(과거
    ±1~1.5℃ → 새 진입/해제 간격(0.25℃) 대비)로 줄여야 동일한 회귀 의도(작은 노이즈에
    채터링 없음)를 유지할 수 있다."""
    base = 23.3  # 진입 문턱(23.0) 바로 위 — 경계 부근
    swing = [0.15, -0.12, 0.13, -0.10, 0.14, -0.11, 0.12, -0.13, 0.11, -0.14]
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
    # 이슈 #51(중앙 유지형 재설계) — 진입 문턱(23.0℃)과 해제 문턱(22.75℃) 사이(22.9℃)를
    # 사용한다: 신규(fresh) 상태에선 진입 문턱 미달로 OFF, 오염(was_high=True) 상태에선
    # 해제 문턱을 넘겨 ON을 유지 — 아래 두 타임라인의 첫 시간 판정이 갈리는 지점.
    normal_baseline = _baseline({h: (22.9, 70.0) for h in range(3)})

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


# ── run_notify — 긴급 시점별 분기(이슈 #38 A안: 현재🚨/미래🔮 요약/과거 무발송) ──
def _hot_forecast(date_str="20260703"):
    """24시간 내내 45℃(밴드 상한을 훨씬 초과) — 냉방 풀가동으로도 못 잡아 이른 시간부터
    늦은 시간까지 emergency_hours가 폭넓게 걸리는 프로파일."""
    hourly = [{"date": date_str, "time": f"{h:02d}00", "temp": 45.0, "humidity": 70.0}
              for h in range(24)]
    return {"unavailable": False, "hourly": hourly, "daily": []}


def test_run_notify_splits_current_future_past_emergency(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """현재 시각(now=12시) 기준 — 과거(hour<12)에 이미 걸린 긴급은 발송 안 함, 현재
    (hour==12)는 개별 🚨 긴급, 미래(hour>12)는 🔮 사전 경보 1건으로 요약 발송돼야 한다."""
    from datetime import date, datetime
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: _hot_forecast())
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    now = datetime(2026, 7, 3, 12, 0)
    live.run_notify(dry_run=False, today=today, now=now)

    emergency_embeds = [e for e in sent if e["title"].startswith("🚨")]
    forecast_embeds = [e for e in sent if e["title"].startswith("🔮")]

    # 과거(hour<12)는 어떤 형태로도 발송되지 않는다 — 🚨 임베드의 "일시" 필드에
    # 12시보다 이른 시각이 없어야 하고, 🔮 임베드는 미래만 담아야 한다.
    for e in emergency_embeds:
        hour_str = e["fields"][0]["value"].split()[-1]  # "HH시"
        assert hour_str == "12시"
    if forecast_embeds:
        assert len(forecast_embeds) == 1
        for f in forecast_embeds[0]["fields"]:
            hour_str = f["name"].split()[-1]
            assert int(hour_str.replace("시", "")) > 12


def test_run_notify_emergency_dedup_no_resend_on_rerun(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """같은 now로 재실행하면 긴급(🚨/🔮) 재발송이 없어야 한다(dedup 키는 hour:kind)."""
    from datetime import date, datetime
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: _hot_forecast())
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    now = datetime(2026, 7, 3, 12, 0)
    live.run_notify(dry_run=False, today=today, now=now)
    n1 = len(sent)
    assert n1 > 0

    live.run_notify(dry_run=False, today=today, now=now)
    n2 = len(sent) - n1
    assert n2 == 0  # 재실행 — 긴급 재발송 0건(장치 전환 임베드는 별개라 총량이 아닌 증가분으로 확인)


def test_run_notify_past_emergency_not_resent_after_now_advances(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """미래였던 시간대가 시간이 흘러 과거가 되면(예: now가 13시→15시로 진행) 그사이
    이미 dedup 키에 등록된 시간대는 재발송되지 않는다(뒷북 제거와 dedup이 함께 성립)."""
    from datetime import date, datetime
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: _hot_forecast())
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    live.run_notify(dry_run=False, today=today, now=datetime(2026, 7, 3, 12, 0))
    before = len(sent)
    # 시간이 흘러 15시 재판정 — 12~15시가 이제 과거로 바뀌어도 이미 상태 키에 남아
    # 있으므로 재발송되지 않아야 한다(신규 미래 구간만 있으면 그만큼만 늘어날 수 있음).
    live.run_notify(dry_run=False, today=today, now=datetime(2026, 7, 3, 15, 0))
    new_emergency_or_forecast = [e for e in sent[before:]
                                  if e["title"].startswith("🚨") or e["title"].startswith("🔮")]
    # 15시(현재) 자체가 이미 이전 실행의 "미래 요약"에 포함돼 dedup 키가 있으므로
    # 신규 긴급성 임베드는 없어야 한다.
    assert new_emergency_or_forecast == []


def test_run_notify_midnight_reset_reallows_emergency_dispatch(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """자정 리셋(날짜가 바뀌면 emergency_hours 상태 초기화) 후에는 같은 시간대라도 다시
    긴급 판정·발송이 가능해야 한다."""
    from datetime import date, datetime
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: _hot_forecast())
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    live.run_notify(dry_run=False, today=date(2026, 7, 3), now=datetime(2026, 7, 3, 12, 0))
    before = len(sent)

    monkeypatch.setattr(weather, "get_forecast_3d", lambda: _hot_forecast(date_str="20260704"))
    live.run_notify(dry_run=False, today=date(2026, 7, 4), now=datetime(2026, 7, 4, 12, 0))
    after_new_day = [e for e in sent[before:]
                      if e["title"].startswith("🚨") or e["title"].startswith("🔮")]
    assert after_new_day  # 자정 리셋 — 같은 프로파일이라도 다시 발송됨


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


# ── record_snapshot / load_today_snapshots / archive_snapshots (이슈 #40) ───────────
def _make_item(hour, ctrl_temp=22.0, ctrl_hum=70.0, events=None):
    from control.actuators import ControlLog
    return {"hour": hour, "out_temp": 26.0, "out_hum": 78.0, "base_temp": 26.7, "base_hum": 88.0,
            "ctrl_temp": ctrl_temp, "ctrl_hum": ctrl_hum, "devices_on": ["dehumidifier"],
            "events": events if events is not None else
            [ControlLog(date="2026-07-05", device="dehumidifier", action="ON", reason="고습")]}


def test_record_snapshot_first_write_wins():
    state = {}
    item = _make_item(14)
    assert live.record_snapshot(state, item, source="sim") is True
    assert state["snapshots"]["14"]["ctrl_temp"] == 22.0
    assert state["version"] == 2

    # 재기록 시도 — 값이 달라도 덮지 않고 False
    item2 = _make_item(14, ctrl_temp=99.0)
    assert live.record_snapshot(state, item2, source="sim") is False
    assert state["snapshots"]["14"]["ctrl_temp"] == 22.0  # 불변


def test_record_snapshot_serializes_control_log_events():
    state = {}
    item = _make_item(9)
    live.record_snapshot(state, item)
    events = state["snapshots"]["9"]["events"]
    assert events == [{"device": "dehumidifier", "action": "ON", "reason": "고습", "mode": "auto"}]


def test_load_today_snapshots_date_mismatch_returns_empty(_isolated_state):
    from datetime import date
    state = {"date": "2026-07-04", "snapshots": {"10": {"ctrl_temp": 1.0}}}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(state), encoding="utf-8")
    assert live.load_today_snapshots(today=date(2026, 7, 5)) == {}


def test_load_today_snapshots_v1_file_backward_compat(_isolated_state):
    """snapshots 키가 없는 v1 상태 파일 — 빈 dict로 자연 흡수(예외 없음)."""
    from datetime import date
    state = {"date": "2026-07-05", "devices": {}, "last_ctrl": None}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(state), encoding="utf-8")
    assert live.load_today_snapshots(today=date(2026, 7, 5)) == {}


def test_load_today_snapshots_matches_date(_isolated_state):
    from datetime import date
    state = {"date": "2026-07-05", "snapshots": {"10": {"ctrl_temp": 1.0}}}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(state), encoding="utf-8")
    assert live.load_today_snapshots(today=date(2026, 7, 5)) == {"10": {"ctrl_temp": 1.0}}


@pytest.fixture
def _isolated_history(tmp_path, monkeypatch):
    history_path = tmp_path / "control_history.json"
    monkeypatch.setattr(live, "HISTORY_PATH", history_path)
    return history_path


def test_archive_snapshots_merges_by_date(_isolated_history):
    prev_state = {"date": "2026-07-04", "snapshots": {"9": {"ctrl_temp": 20.0}}}
    live.archive_snapshots(prev_state)
    history = json.loads(_isolated_history.read_text(encoding="utf-8"))
    assert history == {"2026-07-04": {"9": {"ctrl_temp": 20.0}}}


def test_archive_snapshots_noop_when_no_snapshots(_isolated_history):
    live.archive_snapshots({"date": "2026-07-04", "snapshots": {}})
    assert not _isolated_history.exists()
    live.archive_snapshots({})
    assert not _isolated_history.exists()


def test_archive_snapshots_prunes_to_30_days(_isolated_history):
    import json as _json
    old_history = {f"2026-01-{d:02d}": {"0": {}} for d in range(1, 32)}  # 31일치
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(_json.dumps(old_history), encoding="utf-8")

    prev_state = {"date": "2026-02-01", "snapshots": {"0": {"ctrl_temp": 5.0}}}
    live.archive_snapshots(prev_state)
    history = json.loads(_isolated_history.read_text(encoding="utf-8"))
    assert len(history) == 30
    assert "2026-02-01" in history
    assert "2026-01-01" not in history  # 가장 오래된 것이 프룬됨


# ── load_recent_days_avg (이슈 #57, 대시보드 "최근 N일" 차트) ───────────────────
def test_load_recent_days_avg_averages_hourly_snapshots(_isolated_history):
    history = {
        "2026-07-04": {
            "0": {"out_temp": 20.0, "base_temp": 22.0, "ctrl_temp": 23.0},
            "12": {"out_temp": 30.0, "base_temp": 28.0, "ctrl_temp": 25.0},
        },
    }
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(json.dumps(history), encoding="utf-8")

    from datetime import date
    result = live.load_recent_days_avg(days=7, today=date(2026, 7, 4))
    assert result["2026-07-04"]["out_temp"] == pytest.approx(25.0)
    assert result["2026-07-04"]["base_temp"] == pytest.approx(25.0)
    assert result["2026-07-04"]["ctrl_temp"] == pytest.approx(24.0)


def test_load_recent_days_avg_excludes_none_values_from_average(_isolated_history):
    history = {
        "2026-07-04": {
            "0": {"out_temp": None, "base_temp": 22.0, "ctrl_temp": 20.0},
            "1": {"out_temp": 10.0, "base_temp": None, "ctrl_temp": 24.0},
        },
    }
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(json.dumps(history), encoding="utf-8")

    from datetime import date
    result = live.load_recent_days_avg(days=7, today=date(2026, 7, 4))
    assert result["2026-07-04"]["out_temp"] == pytest.approx(10.0)   # None 제외 후 평균
    assert result["2026-07-04"]["base_temp"] == pytest.approx(22.0)  # None 제외 후 평균
    assert result["2026-07-04"]["ctrl_temp"] == pytest.approx(22.0)


def test_load_recent_days_avg_truncates_to_recent_n_days(_isolated_history):
    history = {f"2026-06-{d:02d}": {"0": {"out_temp": float(d), "base_temp": float(d), "ctrl_temp": float(d)}}
               for d in range(1, 11)}  # 10일치
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(json.dumps(history), encoding="utf-8")

    from datetime import date
    result = live.load_recent_days_avg(days=7, today=date(2026, 6, 10))
    assert len(result) == 7
    assert list(result.keys()) == [f"2026-06-{d:02d}" for d in range(4, 11)]  # 오름차순, 최근 7일


def test_load_recent_days_avg_gap_does_not_pull_in_dates_outside_calendar_window(_isolated_history):
    """리뷰 P2 회귀 방지 — 창(days=7) 안에 서버 다운타임 등으로 결측 구간(중간 3일 전 필드
    None)이 있어도, 개수를 채우려고 캘린더 창(today 기준 최근 7일) 밖의 더 오래된 날짜를
    끌어오면 안 된다. 반환 키는 전부 창 안에 있어야 하고, 결측만큼 7개 미만이어야 한다
    (dashboard.py "기록 축적 중" 캡션이 정상 발동하는 전제)."""
    history = {f"2026-06-{d:02d}": {"0": {"out_temp": float(d), "base_temp": float(d), "ctrl_temp": float(d)}}
               for d in range(1, 11)}  # 06-01 ~ 06-10, 10일치
    for d in (6, 7, 8):  # 창 안(06-04~06-10)의 중간 3일을 전 필드 결측으로 만든다
        history[f"2026-06-{d:02d}"] = {"0": {"out_temp": None, "base_temp": None, "ctrl_temp": None}}
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(json.dumps(history), encoding="utf-8")

    from datetime import date
    today = date(2026, 6, 10)
    result = live.load_recent_days_avg(days=7, today=today)

    window = {f"2026-06-{d:02d}" for d in range(4, 11)}  # today 기준 최근 7 캘린더일(06-04~06-10)
    assert set(result.keys()) <= window          # 창 밖 날짜(06-01~03) 유입 없음
    assert len(result) < 7                        # 결측 3일만큼 자연히 7개 미만
    assert set(result.keys()) == {"2026-06-04", "2026-06-05", "2026-06-09", "2026-06-10"}


def test_load_recent_days_avg_empty_when_no_history_file(_isolated_history):
    from datetime import date
    assert live.load_recent_days_avg(days=7, today=date(2026, 7, 4)) == {}


def test_load_recent_days_avg_includes_today_snapshot(_isolated_history, _isolated_state):
    from datetime import date
    history = {"2026-07-03": {"0": {"out_temp": 20.0, "base_temp": 21.0, "ctrl_temp": 22.0}}}
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(json.dumps(history), encoding="utf-8")

    today_state = {"date": "2026-07-04",
                    "snapshots": {"9": {"out_temp": 26.0, "base_temp": 27.0, "ctrl_temp": 24.0}}}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(today_state), encoding="utf-8")

    result = live.load_recent_days_avg(days=7, today=date(2026, 7, 4))
    assert set(result.keys()) == {"2026-07-03", "2026-07-04"}
    assert result["2026-07-04"]["out_temp"] == pytest.approx(26.0)


def test_load_recent_days_avg_day_with_all_fields_none_excluded(_isolated_history):
    history = {"2026-07-04": {"0": {"out_temp": None, "base_temp": None, "ctrl_temp": None}}}
    _isolated_history.parent.mkdir(parents=True, exist_ok=True)
    _isolated_history.write_text(json.dumps(history), encoding="utf-8")

    from datetime import date
    assert live.load_recent_days_avg(days=7, today=date(2026, 7, 4)) == {}


# ── simulate_control seed_ctrl (이슈 #40) ───────────────────────────────────────
def test_simulate_control_seed_ctrl_applies_start_value():
    baseline = _baseline({h: (30.0, 90.0) for h in range(3)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-05",
                                      seed_ctrl=(21.0, 65.0))
    assert timeline[0]["ctrl_temp"] == pytest.approx(21.0)
    assert timeline[0]["ctrl_hum"] == pytest.approx(65.0)


def test_simulate_control_seed_ctrl_overrides_initial_ctrl():
    from datetime import date as _dt
    baseline = _baseline({h: (30.0, 90.0) for h in range(3)})
    states = default_states()
    yesterday_ctrl = {"date": "2026-07-04", "temp": 10.0, "hum": 10.0}
    timeline = live.simulate_control(baseline, _sp(), states, date=_dt(2026, 7, 5),
                                      initial_ctrl=yesterday_ctrl, seed_ctrl=(21.0, 65.0))
    assert timeline[0]["ctrl_temp"] == pytest.approx(21.0)
    assert timeline[0]["ctrl_hum"] == pytest.approx(65.0)


def test_simulate_control_seed_ctrl_none_keeps_existing_behavior():
    baseline = _baseline({h: (30.0, 90.0) for h in range(3)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-05", seed_ctrl=None)
    assert timeline[0]["ctrl_temp"] == pytest.approx(30.0)  # 기존 거동(기준선 그대로)


# ── assemble_today_timeline (이슈 #40) ──────────────────────────────────────────
def _outdoor(hours_temp: dict, hum: float = 70.0):
    return [{"hour": h, "temp": t, "humidity": hum} for h, t in hours_temp.items()]


def test_assemble_today_timeline_past_from_snapshot_future_from_simulation(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    today = date(2026, 7, 5)

    state = {"date": "2026-07-05", "snapshots": {
        "8": {"out_temp": 20.0, "out_hum": 70.0, "base_temp": 20.0, "base_hum": 70.0,
              "ctrl_temp": 21.0, "ctrl_hum": 65.0, "devices_on": [], "events": [],
              "source": "sim", "recorded_at": "2026-07-05T08:00:00"},
    }}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(state), encoding="utf-8")

    outdoor = _outdoor({h: 22.0 for h in range(6, 24)})
    states = default_states()
    timeline = live.assemble_today_timeline(outdoor, _sp(), states, today, now_hour=10)

    hours = [t["hour"] for t in timeline]
    assert 8 in hours  # 과거=스냅샷 복원
    assert timeline[hours.index(8)]["ctrl_temp"] == pytest.approx(21.0)
    assert 10 in hours and max(hours) >= 10  # 현재·미래=시뮬 합성 포함
    # 경계 연속성 — 미래 첫 시간(now_hour=10)의 ctrl_temp가 스냅샷 마지막 값(21.0)에서 이어짐
    fut_first = next(t for t in timeline if t["hour"] == 10)
    assert fut_first["ctrl_temp"] == pytest.approx(21.0)


def test_assemble_today_timeline_no_snapshots_matches_legacy_path(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """스냅샷이 하나도 없으면 기존 경로(load_last_ctrl + fallback_clamp=True)와 동일 결과."""
    from datetime import date
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    today = date(2026, 7, 5)
    outdoor = _outdoor({h: 30.0 for h in range(24)})

    states1 = default_states()
    legacy_baseline = live.indoor_baseline(outdoor, date=today)
    legacy = live.simulate_control(legacy_baseline, _sp(), states1, date=today,
                                    initial_ctrl=live.load_last_ctrl(), fallback_clamp=True)

    states2 = default_states()
    assembled = live.assemble_today_timeline(outdoor, _sp(), states2, today, now_hour=0)

    assert [t["hour"] for t in assembled] == [t["hour"] for t in legacy]
    assert assembled[0]["ctrl_temp"] == pytest.approx(legacy[0]["ctrl_temp"])


def test_assemble_today_timeline_missing_past_hour_falls_back_to_simulation(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """기록 없는 과거 시간이 outdoor에 남아 있으면 시뮬값으로 폴백해 결측 없이 채운다."""
    from datetime import date
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    today = date(2026, 7, 5)
    outdoor = _outdoor({h: 22.0 for h in range(6, 24)})  # 6시부터 데이터 있음, 스냅샷은 없음
    states = default_states()
    timeline = live.assemble_today_timeline(outdoor, _sp(), states, today, now_hour=10)
    hours = [t["hour"] for t in timeline]
    assert 8 in hours  # 과거(6~9시)도 시뮬값 폴백으로 채워짐 — 결측 아님
    assert 6 in hours


def test_assemble_today_timeline_missing_past_hour_no_outdoor_is_omitted(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """기록도 없고 outdoor에도 없는 과거 시간은 결측 생략(행 없음)."""
    from datetime import date
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    today = date(2026, 7, 5)
    outdoor = _outdoor({h: 22.0 for h in range(10, 24)})  # 10시 이전 데이터 없음
    states = default_states()
    timeline = live.assemble_today_timeline(outdoor, _sp(), states, today, now_hour=10)
    hours = [t["hour"] for t in timeline]
    assert 5 not in hours
    assert 0 not in hours


def test_assemble_today_timeline_outdoor_none_returns_only_past_snapshots(_isolated_state):
    from datetime import date
    today = date(2026, 7, 5)
    state = {"date": "2026-07-05", "snapshots": {
        "8": {"out_temp": 20.0, "out_hum": 70.0, "base_temp": 20.0, "base_hum": 70.0,
              "ctrl_temp": 21.0, "ctrl_hum": 65.0, "devices_on": [], "events": []},
        "12": {"out_temp": 20.0, "out_hum": 70.0, "base_temp": 20.0, "base_hum": 70.0,
               "ctrl_temp": 21.0, "ctrl_hum": 65.0, "devices_on": [], "events": []},
    }}
    _isolated_state.parent.mkdir(parents=True, exist_ok=True)
    _isolated_state.write_text(json.dumps(state), encoding="utf-8")

    states = default_states()
    timeline = live.assemble_today_timeline(None, _sp(), states, today, now_hour=10)
    hours = [t["hour"] for t in timeline]
    assert hours == [8]  # 12시는 now_hour(10) 이후라 과거 아님 → 미포함, outdoor 없어 미래도 없음


def test_assemble_today_timeline_outdoor_none_no_snapshots_returns_empty(_isolated_state):
    from datetime import date
    states = default_states()
    timeline = live.assemble_today_timeline(None, _sp(), states, date(2026, 7, 5), now_hour=10)
    assert timeline == []


# ── run_notify — 스냅샷 기록·보존·롤오버(이슈 #40) ───────────────────────────────
def test_run_notify_records_snapshot_for_current_hour(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    _patch_notify(monkeypatch, [])

    today = date(2026, 7, 3)
    now = datetime(2026, 7, 3, 12, 0)
    live.run_notify(dry_run=False, today=today, now=now)

    state = json.loads(_isolated_state.read_text(encoding="utf-8"))
    assert state["version"] == 2
    assert "12" in state["snapshots"]
    assert state["snapshots"]["12"]["source"] == "sim"


def test_run_notify_accumulates_snapshots_same_day(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    _patch_notify(monkeypatch, [])

    today = date(2026, 7, 3)
    live.run_notify(dry_run=False, today=today, now=datetime(2026, 7, 3, 9, 0))
    live.run_notify(dry_run=False, today=today, now=datetime(2026, 7, 3, 12, 0))

    state = json.loads(_isolated_state.read_text(encoding="utf-8"))
    assert "9" in state["snapshots"]
    assert "12" in state["snapshots"]  # 누적


def test_run_notify_kma_fail_preserves_snapshots(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    _patch_notify(monkeypatch, [])

    today = date(2026, 7, 3)
    live.run_notify(dry_run=False, today=today, now=datetime(2026, 7, 3, 9, 0))

    # 두 번째 호출은 KMA 실패
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: {"unavailable": True, "reason": "일시 오류"})
    live.run_notify(dry_run=False, today=today, now=datetime(2026, 7, 3, 10, 0))

    state = json.loads(_isolated_state.read_text(encoding="utf-8"))
    assert "9" in state["snapshots"]  # KMA 실패에도 이전 스냅샷 유실 없음


def test_run_notify_date_rollover_archives_and_prunes(
        monkeypatch, _isolated_state, _isolated_setpoints, _isolated_history):
    from datetime import date, datetime
    from llm import weather
    ramp = {h: 20.0 + h * 1.0 for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)
    _patch_notify(monkeypatch, [])

    live.run_notify(dry_run=False, today=date(2026, 7, 3), now=datetime(2026, 7, 3, 12, 0))

    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(ramp, date_str="20260704"))
    live.run_notify(dry_run=False, today=date(2026, 7, 4), now=datetime(2026, 7, 4, 9, 0))

    history = json.loads(_isolated_history.read_text(encoding="utf-8"))
    assert "2026-07-03" in history
    assert "12" in history["2026-07-03"]

    new_state = json.loads(_isolated_state.read_text(encoding="utf-8"))
    assert new_state["date"] == "2026-07-04"
    assert "12" not in new_state.get("snapshots", {})  # 새로 시작(전날 것 안 이어받음)
