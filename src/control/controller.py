"""규칙 기반 자동 제어 판정 — decide()·emergency() 순수 함수(LLM 호출 없음)."""
from .actuators import ControlLog, DeviceState


def _band_state(value: float, low: float, high: float, deadband: float,
                 was_high: bool, was_low: bool) -> str:
    """히스테리시스 밴드 판정 — "high"/"low"/"normal".

    ON(진입)은 밴드 밖에서만, 지속(was_high/was_low)은 데드밴드 안쪽까지 유지하다
    데드밴드를 넘어 안쪽으로 복귀해야 비로소 "normal"(OFF)로 전환(채터링 방지).
    """
    if value > high:
        return "high"
    if value < low:
        return "low"
    if was_high and value > high - deadband:
        return "high"
    if was_low and value < low + deadband:
        return "low"
    return "normal"


def decide(reading: dict, setpoints, states: dict[str, DeviceState], date=None) -> list[ControlLog]:
    """센서값+설정 밴드+현재 장치 상태 → 상태 변화(states 갱신) + 발생한 ControlLog 목록.

    수동(auto=False) 장치는 결정 대상에서 제외(현재 상태 그대로 유지, 로그 없음).
    heater/cooling_fan 동시 ON 금지 — 충돌 시 이탈 폭이 큰 쪽 우선.
    """
    temp = reading.get("온도내부_평균", (setpoints.temp_low + setpoints.temp_high) / 2)
    hum = reading.get("습도내부_평균", (setpoints.hum_low + setpoints.hum_high) / 2)
    logs: list[ControlLog] = []

    def cur_on(device: str) -> bool:
        return states[device].on

    # vent는 온도·습도 두 규칙이 공유하는 장치라, 습도측 히스테리시스(was_high)는 vent가
    # "습도 사유"로 켜져 있을 때만 인정한다 — 온도 사유로 켜진 vent를 습도 데드밴드 유지로
    # 오인하지 않도록(이슈 #17 P2-2).
    vent_was_high_hum = cur_on("vent") and states["vent"].cause == "hum"

    temp_state = _band_state(temp, setpoints.temp_low, setpoints.temp_high,
                              setpoints.temp_deadband, cur_on("cooling_fan"), cur_on("heater"))
    hum_state = _band_state(hum, setpoints.hum_low, setpoints.hum_high,
                             setpoints.hum_deadband, vent_was_high_hum, cur_on("humidifier"))

    want_cooling = temp_state == "high"
    want_heater = temp_state == "low"
    if want_cooling and want_heater:  # 이론상 low<high면 불가하지만 방어적으로 처리
        if (temp - setpoints.temp_high) >= (setpoints.temp_low - temp):
            want_heater = False
        else:
            want_cooling = False

    # vent: 온도 상한 초과 시 강제 ON, 온도 하한 미달 시 강제 OFF(스펙 명시),
    # 온도가 정상이면 습도 판정에 위임.
    if temp_state == "high":
        want_vent, vent_cause = True, "temp"
        vent_reason = f"온도 상한 초과({temp:.1f}℃>{setpoints.temp_high:.1f}℃) — 환기 가동"
    elif temp_state == "low":
        want_vent, vent_cause = False, None
        vent_reason = "온도 하한 미달 — 난방 우선(환기 정지)"
    else:
        want_vent = hum_state == "high"
        vent_cause = "hum" if want_vent else None
        vent_reason = (f"습도 상한 초과({hum:.1f}%>{setpoints.hum_high:.1f}%) — 환기 가동"
                        if want_vent else "밴드 정상 범위 복귀")

    want_humidifier = hum_state == "low"

    def _apply(device: str, want: bool, reason: str, cause: str | None = None):
        state = states[device]
        if not state.auto:
            return
        if state.on == want:
            if want:  # 지속 ON — cause만 최신화(온도↔습도 전환 대비)
                state.cause = cause
            return
        state.on = want
        state.cause = cause if want else None
        logs.append(ControlLog(date=str(date), device=device, action="ON" if want else "OFF",
                                reason=reason, mode="auto"))

    _apply("cooling_fan", want_cooling,
           f"온도 상한 초과({temp:.1f}℃>{setpoints.temp_high:.1f}℃)" if want_cooling
           else "밴드 정상 범위 복귀")
    _apply("heater", want_heater,
           f"온도 하한 미달({temp:.1f}℃<{setpoints.temp_low:.1f}℃)" if want_heater
           else "밴드 정상 범위 복귀")
    _apply("vent", want_vent, vent_reason, cause=vent_cause)
    _apply("humidifier", want_humidifier,
           f"습도 하한 미달({hum:.1f}%<{setpoints.hum_low:.1f}%)" if want_humidifier
           else "밴드 정상 범위 복귀")

    return logs


def akey(alert: dict) -> str:
    """monitor._akey 와 동일한 dedup 식별자 규칙 — key:level."""
    return f"{alert['key']}:{alert['level']}"


def _emergency_candidates(recent_readings: list[dict], setpoints,
                           states: dict[str, DeviceState]) -> list[dict]:
    """최근 3틱 연속 밴드 밖 + 관련 장치 풀가동 조건을 만족하는 후보 alert 전부(없으면 [])."""
    if len(recent_readings) < 3:
        return []
    last3 = recent_readings[-3:]
    temps = [r.get("온도내부_평균") for r in last3]
    hums = [r.get("습도내부_평균") for r in last3]

    candidates = []
    if all(t is not None and t > setpoints.temp_high for t in temps) \
            and states["cooling_fan"].on and states["vent"].on:
        candidates.append({"key": "control_limit:temp_high", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(냉방·환기 풀가동에도 고온 지속)"})
    if all(t is not None and t < setpoints.temp_low for t in temps) and states["heater"].on:
        candidates.append({"key": "control_limit:temp_low", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(난방 풀가동에도 저온 지속)"})
    if all(h is not None and h > setpoints.hum_high for h in hums) and states["vent"].on:
        candidates.append({"key": "control_limit:hum_high", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(환기 풀가동에도 고습 지속)"})
    if all(h is not None and h < setpoints.hum_low for h in hums) and states["humidifier"].on:
        candidates.append({"key": "control_limit:hum_low", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(가습 풀가동에도 저습 지속)"})
    return candidates


def emergency(recent_readings: list[dict], setpoints, states: dict[str, DeviceState],
              active: set | None = None) -> tuple[list[dict], set]:
    """긴급 경보 판정 — src/llm/monitor.evaluate()와 동일한 자기 정리(self-cleaning) dedup 패턴.

    반환: (신규로 진입한 alert 목록, 현재 조건을 만족하는 키셋). 호출부는 매 틱
    active를 반환된 키셋으로 **교체**해야 한다(add만 하면 조건 해소 후에도 영구 억제됨 —
    이슈 #17 P1-1). 키는 control_limit:temp_high/temp_low/hum_high/hum_low로 분리해
    상한·하한이 같은 키를 공유해 서로를 지우는 문제를 방지한다(P1-2).
    """
    active = active or set()
    candidates = _emergency_candidates(recent_readings, setpoints, states)
    keys = {akey(a) for a in candidates}
    to_send = [a for a in candidates if akey(a) not in active]
    return to_send, keys
