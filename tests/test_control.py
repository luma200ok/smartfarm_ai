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
def test_temp_above_high_turns_on_cooling_heater_off():
    states = default_states()
    states["heater"].on = True  # 이전에 켜져 있던 히터도 강제 OFF 확인
    logs = decide(_reading(temp=26.0), _sp(), states, date="2024-01-01")
    assert states["cooling_fan"].on is True
    assert states["heater"].on is False
    actions = {log.device: log.action for log in logs}
    assert actions["cooling_fan"] == "ON"
    assert actions["heater"] == "OFF"


def test_temp_below_low_turns_on_heater_and_off_cooling():
    states = default_states()
    states["cooling_fan"].on = True
    logs = decide(_reading(temp=19.0), _sp(), states, date="2024-01-01")
    assert states["heater"].on is True
    assert states["cooling_fan"].on is False
    assert any(log.device == "heater" and log.action == "ON" for log in logs)


def test_temp_within_band_no_temp_device_change():
    states = default_states()
    logs = decide(_reading(temp=22.0, hum=70.0), _sp(), states, date="2024-01-01")
    assert states["cooling_fan"].on is False
    assert states["heater"].on is False
    assert logs == []  # 이미 OFF 상태 유지 → 로그 없음


def test_hum_above_high_turns_on_dehumidifier():
    states = default_states()
    logs = decide(_reading(temp=22.0, hum=90.0), _sp(), states, date="2024-01-01")
    assert states["dehumidifier"].on is True
    assert any(log.device == "dehumidifier" and log.action == "ON" for log in logs)


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


# ── 습도 밴드 중앙 목표 OFF 판정(이슈 #33) ───────────────────────────────
def test_hum_dehumidifier_stays_on_until_mid_reached():
    """hum_mode="center"(live 경로 전용, 이슈 #33) — 제습기 ON 후 밴드 안(85% 아래)이어도
    중앙(72.5%)에서 데드밴드(2.0) 밖이면 유지."""
    states = default_states()
    decide(_reading(hum=90.0), _sp(), states, date="d1", hum_mode="center")
    assert states["dehumidifier"].on is True
    logs = decide(_reading(hum=80.0), _sp(), states, date="d2", hum_mode="center")  # 밴드 안, 중앙과 거리 7.5 > 2.0
    assert states["dehumidifier"].on is True
    assert logs == []


def test_hum_dehumidifier_turns_off_near_mid():
    """hum_mode="center" — 중앙(72.5%) 데드밴드(2.0) 이내로 들어오면 OFF."""
    states = default_states()
    decide(_reading(hum=90.0), _sp(), states, date="d1", hum_mode="center")
    assert states["dehumidifier"].on is True
    logs = decide(_reading(hum=73.0), _sp(), states, date="d2", hum_mode="center")  # |73-72.5|=0.5 <= 2.0
    assert states["dehumidifier"].on is False
    assert any(log.device == "dehumidifier" and log.action == "OFF" for log in logs)


def test_hum_humidifier_turns_off_near_mid_symmetric():
    """hum_mode="center" — 가습기도 동일. 중앙 근접 시 OFF(저습 대칭)."""
    states = default_states()
    decide(_reading(hum=50.0), _sp(), states, date="d1", hum_mode="center")
    assert states["humidifier"].on is True
    logs = decide(_reading(hum=72.0), _sp(), states, date="d2", hum_mode="center")  # |72-72.5|=0.5 <= 2.0
    assert states["humidifier"].on is False
    assert any(log.device == "humidifier" and log.action == "OFF" for log in logs)


def test_hum_edge_mode_is_default_and_matches_temp_hysteresis():
    """hum_mode 기본값(edge)은 온도와 동일한 경계 히스테리시스(_band_state)를 쓴다 —
    리플레이(app/views/monitor.py._run_control_step)는 이 기본값을 그대로 사용해야 하므로
    회귀 방지용으로 명시 검증(이슈 #33 리뷰 P1 픽스)."""
    states = default_states()
    decide(_reading(hum=90.0), _sp(), states, date="d1")  # hum_mode 미지정
    assert states["dehumidifier"].on is True
    # 밴드 안(85% 아래)이라도 경계 히스테리시스 창(high-deadband=83.0) 안쪽까지 와야 OFF —
    # 중앙(72.5%)까지 갈 필요 없음(edge≠center 확인).
    logs = decide(_reading(hum=84.0), _sp(), states, date="d2")  # 83.0보다 큼 → 유지
    assert states["dehumidifier"].on is True
    assert logs == []
    logs2 = decide(_reading(hum=82.0), _sp(), states, date="d3")  # 83.0보다 작음 → OFF
    assert states["dehumidifier"].on is False
    assert any(log.device == "dehumidifier" and log.action == "OFF" for log in logs2)


# ── 온도 밴드 중앙 목표 OFF 판정(이슈 #45, 습도와 대칭) ───────────────────
def test_temp_cooling_stays_on_until_mid_reached():
    """temp_mode="center"(live 경로 전용, 이슈 #45) — 냉방 ON 후 밴드 안(25℃ 아래)이어도
    중앙(22.5℃)에서 데드밴드(0.5) 밖이면 유지."""
    states = default_states()
    decide(_reading(temp=26.0), _sp(), states, date="d1", temp_mode="center")
    assert states["cooling_fan"].on is True
    logs = decide(_reading(temp=24.0), _sp(), states, date="d2", temp_mode="center")  # 밴드 안, 중앙과 거리 1.5 > 0.5
    assert states["cooling_fan"].on is True
    assert logs == []


def test_temp_cooling_turns_off_near_mid():
    """temp_mode="center" — 중앙(22.5℃) 데드밴드(0.5) 이내로 들어오면 OFF."""
    states = default_states()
    decide(_reading(temp=26.0), _sp(), states, date="d1", temp_mode="center")
    assert states["cooling_fan"].on is True
    logs = decide(_reading(temp=23.0), _sp(), states, date="d2", temp_mode="center")  # |23-22.5|=0.5 <= 0.5
    assert states["cooling_fan"].on is False
    assert any(log.device == "cooling_fan" and log.action == "OFF" for log in logs)


def test_temp_heater_turns_off_near_mid_symmetric():
    """temp_mode="center" — 히터도 동일. 중앙 근접 시 OFF(저온 대칭)."""
    states = default_states()
    decide(_reading(temp=19.0), _sp(), states, date="d1", temp_mode="center")
    assert states["heater"].on is True
    logs = decide(_reading(temp=22.0), _sp(), states, date="d2", temp_mode="center")  # |22-22.5|=0.5 <= 0.5
    assert states["heater"].on is False
    assert any(log.device == "heater" and log.action == "OFF" for log in logs)


def test_temp_edge_mode_is_default_and_matches_hum_hysteresis():
    """temp_mode 기본값(edge)은 습도와 동일한 경계 히스테리시스(_band_state)를 쓴다 —
    리플레이(app/views/monitor.py._run_control_step)는 이 기본값을 그대로 사용해야 하므로
    회귀 방지용으로 명시 검증(이슈 #45, hum 기본값 검증과 대칭)."""
    states = default_states()
    decide(_reading(temp=26.0), _sp(), states, date="d1")  # temp_mode 미지정
    assert states["cooling_fan"].on is True
    # 밴드 안(25℃ 아래)이라도 경계 히스테리시스 창(high-deadband=24.5) 안쪽까지 와야 OFF —
    # 중앙(22.5℃)까지 갈 필요 없음(edge≠center 확인).
    logs = decide(_reading(temp=24.6), _sp(), states, date="d2")  # 24.5보다 큼 → 유지
    assert states["cooling_fan"].on is True
    assert logs == []
    logs2 = decide(_reading(temp=24.4), _sp(), states, date="d3")  # 24.5보다 작음 → OFF
    assert states["cooling_fan"].on is False
    assert any(log.device == "cooling_fan" and log.action == "OFF" for log in logs2)


# ── 리플레이(edge 모드) + EFFECTS_DAILY 고정 스텝 회귀(이슈 #33 리뷰 P1 픽스) ────────
def test_replay_edge_mode_humidifier_no_overshoot_with_daily_step():
    """리뷰어 재현 시나리오 — 초기 55%(밴드 60~85% 하한 미달)에서 decide(hum_mode="edge",
    기본값)+EFFECTS_DAILY 고정 스텝(+5.0%p)을 반복 적용해도, 가습기가 밴드 상한(85%)을
    관통하지 않고 경계 히스테리시스(하한+deadband=62.0) 안쪽에서 OFF돼야 한다."""
    from control.effects import EFFECTS_DAILY

    sp = _sp()
    states = default_states()
    hum = 55.0
    max_hum_seen = hum

    for step in range(10):
        logs = decide(_reading(hum=hum), sp, states, date=f"d{step}")  # hum_mode 미지정=edge
        if states["humidifier"].on:
            hum += EFFECTS_DAILY["humidifier"]["습도내부_평균"]  # +5.0%p 고정 스텝
        max_hum_seen = max(max_hum_seen, hum)
        if not states["humidifier"].on and step > 0:
            break

    assert states["humidifier"].on is False
    assert states["dehumidifier"].on is False  # 반대쪽 밴드까지 관통해 제습기가 켜지지 않음
    assert max_hum_seen <= sp.hum_high  # 밴드 상한(85%) 관통 없음(회귀 재현 시 90%까지 관통했었음)


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


# ── 충돌 금지: dehumidifier/humidifier 동시 ON 금지(이슈 #27) ────────────
def test_dehumidifier_and_humidifier_never_both_on():
    states = default_states()
    decide(_reading(hum=95.0), _sp(), states, date="d1")
    assert not (states["dehumidifier"].on and states["humidifier"].on)
    decide(_reading(hum=10.0), _sp(), states, date="d2")
    assert not (states["dehumidifier"].on and states["humidifier"].on)


# ── 효과 피드백 ─────────────────────────────────────────────────────────
class _FakeVS:
    """VirtualSensor.inject() 만 필요한 최소 더블."""

    def __init__(self):
        self.calls = []

    def inject(self, feature, start, days, delta, tag="scenario"):
        self.calls.append((feature, start, days, delta))


def test_apply_effects_only_for_on_devices():
    states = default_states()
    states["heater"].on = True
    vs = _FakeVS()
    apply_effects(vs, states, start=5, days=1)
    features = {c[0] for c in vs.calls}
    assert "온도내부_평균" in features
    assert all(c[3] > 0 for c in vs.calls if c[0] == "온도내부_평균")  # heater=+delta
    # cooling_fan/dehumidifier/humidifier 는 OFF → 호출 없음
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
    readings = [_reading(temp=30.0)] * 2
    to_send, keys = emergency(readings, _sp(), states)
    assert to_send == [] and keys == set()


def test_emergency_triggers_after_3_consecutive_ticks_full_power():
    states = default_states()
    states["cooling_fan"].on = True
    readings = [_reading(temp=30.0)] * 3
    to_send, keys = emergency(readings, _sp(), states)
    assert len(to_send) == 1
    assert to_send[0]["level"] == "경고"
    assert to_send[0]["key"] == "control_limit:temp_high"
    assert keys == {"control_limit:temp_high:경고"}


def test_emergency_none_when_device_not_full_power():
    states = default_states()
    states["cooling_fan"].on = False  # 풀가동 아님
    readings = [_reading(temp=30.0)] * 3
    to_send, keys = emergency(readings, _sp(), states)
    assert to_send == [] and keys == set()


def test_emergency_temp_high_and_low_use_distinct_keys():
    """상한 초과·하한 미달은 별개 dedup 키를 쓴다(이슈 #17 P1-2 — 키 충돌 방지)."""
    states = default_states()
    states["heater"].on = True
    readings = [_reading(temp=10.0)] * 3
    to_send, keys = emergency(readings, _sp(), states)
    assert to_send[0]["key"] == "control_limit:temp_low"
    assert keys == {"control_limit:temp_low:경고"}


def test_emergency_dedup_suppresses_repeat_while_condition_persists():
    states = default_states()
    states["cooling_fan"].on = True
    readings = [_reading(temp=30.0)] * 3
    to_send, active = emergency(readings, _sp(), states)
    assert len(to_send) == 1

    to_send2, active2 = emergency(readings, _sp(), states, active=active)
    assert to_send2 == []             # 지속 — 재발송 없음
    assert active2 == active


def test_emergency_self_cleaning_resends_after_condition_recurs():
    """조건이 해소(active에서 제거)됐다가 다시 발생하면 재발송돼야 한다(이슈 #17 P1-2 —
    add-only active set이 영구 억제하던 버그 회귀 확인)."""
    states = default_states()
    states["cooling_fan"].on = True
    hot = [_reading(temp=30.0)] * 3
    to_send, active = emergency(hot, _sp(), states)
    assert len(to_send) == 1

    # 조건 해소(정상 범위 복귀) — 호출부는 active를 반환된 keys로 "교체"해야 한다
    normal = [_reading(temp=22.0)] * 3
    to_send2, active2 = emergency(normal, _sp(), states, active=active)
    assert to_send2 == [] and active2 == set()   # 해소 → active 비워짐(self-cleaning)

    # 재발 — active2(빈 집합) 기준이므로 다시 발송돼야 함
    to_send3, active3 = emergency(hot, _sp(), states, active=active2)
    assert len(to_send3) == 1
    assert active3 == {"control_limit:temp_high:경고"}


# ── P1-1: 시나리오 재적용이 제어 효과 주입을 지우지 않음(실제 VirtualSensor) ──
def test_control_effect_survives_scenario_reapplication(monkeypatch):
    import pandas as pd
    from dl import infer
    from sim import virtual_sensor as vsmod

    def _rows(n=12, year=2024):
        return [{"도": "A", "시군": "B", "농가명": "C", "작기": 1, "품목": "방울토마토",
                  "연도": year, "날짜": f"{year}-01-{i + 1:02d}",
                  **{f: 20.0 for f in infer.ENV_FEATURES}}
                for i in range(n)]

    monkeypatch.setattr(vsmod, "_tomato_df", lambda: pd.DataFrame(_rows()))
    vs = vsmod.VirtualSensor(2024)

    states = default_states()
    states["heater"].on = True
    apply_effects(vs, states, start=vs.cursor, days=1)   # "control" 태그 주입(+1.5℃)

    before = vs.reading()["온도내부_평균"]
    assert before == 20.0 + 1.5   # 효과가 반영돼 있어야 함(사전조건)

    vsmod.apply_scenario(vs, "정상")   # 시나리오 재적용(선택은 그대로 "정상")

    after = vs.reading()["온도내부_평균"]
    assert after == before        # 제어 효과가 지워지지 않아야 함(이슈 #17 P1-1)
