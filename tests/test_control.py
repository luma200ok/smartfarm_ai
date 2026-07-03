"""src/control/ — 규칙 기반 자동 제어(이슈 #17) 테스트."""
import pytest

from control.actuators import default_states
from control.controller import decide, emergency
from control.effects import apply_effects
from control.setpoints import Setpoints


def _sp():
    return Setpoints()  # 온도 20~25℃, 습도 60~85%, 데드밴드 0.5℃/2.0%p


def _reading(temp=22.0, hum=70.0):
    return {"온도내부_평균": temp, "습도내부_평균": hum}


# ── controller.decide — 경계값 ──────────────────────────────────────────
def test_temp_above_high_turns_on_cooling_and_vent_heater_off():
    states = default_states()
    states["heater"].on = True  # 이전에 켜져 있던 히터도 강제 OFF 확인
    logs = decide(_reading(temp=26.0), _sp(), states, date="2024-01-01")
    assert states["cooling_fan"].on is True
    assert states["vent"].on is True
    assert states["heater"].on is False
    actions = {log.device: log.action for log in logs}
    assert actions["cooling_fan"] == "ON"
    assert actions["vent"] == "ON"
    assert actions["heater"] == "OFF"


def test_temp_below_low_turns_on_heater_and_off_cooling_vent():
    states = default_states()
    states["cooling_fan"].on = True
    states["vent"].on = True
    logs = decide(_reading(temp=19.0), _sp(), states, date="2024-01-01")
    assert states["heater"].on is True
    assert states["cooling_fan"].on is False
    assert states["vent"].on is False
    assert any(log.device == "heater" and log.action == "ON" for log in logs)


def test_temp_within_band_no_temp_device_change():
    states = default_states()
    logs = decide(_reading(temp=22.0, hum=70.0), _sp(), states, date="2024-01-01")
    assert states["cooling_fan"].on is False
    assert states["heater"].on is False
    assert logs == []  # 이미 OFF 상태 유지 → 로그 없음


def test_hum_above_high_turns_on_vent():
    states = default_states()
    logs = decide(_reading(temp=22.0, hum=90.0), _sp(), states, date="2024-01-01")
    assert states["vent"].on is True
    assert any(log.device == "vent" and log.action == "ON" for log in logs)


def test_hum_below_low_turns_on_humidifier():
    states = default_states()
    logs = decide(_reading(temp=22.0, hum=50.0), _sp(), states, date="2024-01-01")
    assert states["humidifier"].on is True
    assert any(log.device == "humidifier" and log.action == "ON" for log in logs)


# ── 히스테리시스(데드밴드) ───────────────────────────────────────────────
def test_hysteresis_cooling_stays_on_just_inside_band_no_off():
    """상한 25.0 초과로 ON된 뒤 25.0 바로 아래(데드밴드 0.5 안쪽)로 복귀해도 OFF 안 됨."""
    states = default_states()
    decide(_reading(temp=26.0), _sp(), states, date="d1")
    assert states["cooling_fan"].on is True
    logs = decide(_reading(temp=24.8), _sp(), states, date="d2")  # 25.0 - 0.5 = 24.5 보다 큼
    assert states["cooling_fan"].on is True
    assert logs == []


def test_hysteresis_cooling_turns_off_after_deadband_recovery():
    """데드밴드(24.5) 안쪽까지 복귀하면 OFF."""
    states = default_states()
    decide(_reading(temp=26.0), _sp(), states, date="d1")
    assert states["cooling_fan"].on is True
    logs = decide(_reading(temp=24.0), _sp(), states, date="d2")  # < 24.5
    assert states["cooling_fan"].on is False
    assert any(log.device == "cooling_fan" and log.action == "OFF" for log in logs)


# ── 수동 오버라이드 제외 ────────────────────────────────────────────────
def test_manual_device_excluded_from_auto_decision():
    states = default_states()
    states["cooling_fan"].auto = False
    states["cooling_fan"].on = False
    logs = decide(_reading(temp=30.0), _sp(), states, date="d1")
    assert states["cooling_fan"].on is False  # 수동 유지 — 자동으로 안 바뀜
    assert not any(log.device == "cooling_fan" for log in logs)


# ── 충돌 금지: heater/cooling_fan 동시 ON 금지 ───────────────────────────
def test_heater_and_cooling_never_both_on():
    states = default_states()
    decide(_reading(temp=30.0), _sp(), states, date="d1")
    assert not (states["cooling_fan"].on and states["heater"].on)
    decide(_reading(temp=10.0), _sp(), states, date="d2")
    assert not (states["cooling_fan"].on and states["heater"].on)


# ── 효과 피드백 ─────────────────────────────────────────────────────────
class _FakeVS:
    """VirtualSensor.inject() 만 필요한 최소 더블."""

    def __init__(self):
        self.calls = []

    def inject(self, feature, start, days, delta):
        self.calls.append((feature, start, days, delta))


def test_apply_effects_only_for_on_devices():
    states = default_states()
    states["heater"].on = True
    vs = _FakeVS()
    apply_effects(vs, states, start=5, days=1)
    features = {c[0] for c in vs.calls}
    assert "온도내부_평균" in features
    assert all(c[3] > 0 for c in vs.calls if c[0] == "온도내부_평균")  # heater=+delta
    # cooling_fan/vent/humidifier 는 OFF → 호출 없음
    assert len(vs.calls) == len(set(vs.calls))


def test_apply_effects_off_device_no_injection():
    states = default_states()  # 전부 OFF
    vs = _FakeVS()
    apply_effects(vs, states, start=0, days=1)
    assert vs.calls == []


# ── emergency ───────────────────────────────────────────────────────────
def test_emergency_none_when_less_than_3_ticks():
    states = default_states()
    states["cooling_fan"].on = True
    states["vent"].on = True
    readings = [_reading(temp=30.0)] * 2
    assert emergency(readings, _sp(), states) is None


def test_emergency_triggers_after_3_consecutive_ticks_full_power():
    states = default_states()
    states["cooling_fan"].on = True
    states["vent"].on = True
    readings = [_reading(temp=30.0)] * 3
    alert = emergency(readings, _sp(), states)
    assert alert is not None
    assert alert["level"] == "경고"
    assert alert["key"] == "control_limit:temp"


def test_emergency_none_when_device_not_full_power():
    states = default_states()
    states["cooling_fan"].on = True
    states["vent"].on = False  # 풀가동 아님
    readings = [_reading(temp=30.0)] * 3
    assert emergency(readings, _sp(), states) is None


def test_emergency_dedup_suppresses_repeat():
    states = default_states()
    states["cooling_fan"].on = True
    states["vent"].on = True
    readings = [_reading(temp=30.0)] * 3
    alert = emergency(readings, _sp(), states)
    active = {"control_limit:temp:경고"}
    assert emergency(readings, _sp(), states, active=active) is None
    assert alert is not None
