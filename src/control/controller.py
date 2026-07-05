"""규칙 기반 자동 제어 판정 — decide()·emergency() 순수 함수(LLM 호출 없음).

이슈 #27 — 환기(vent) 제거·제습기(dehumidifier) 신설로 온도(히터/쿨링팬)·습도
(가습기/제습기) 각 2종 전용 장치 구조로 단순화. 습도 장치가 온도 규칙과 더 이상
공유되지 않아 이슈 #17 P2-2의 cause 오염 방지 로직은 불필요해져 제거했다."""
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


def _hum_band_state(value: float, low: float, high: float, deadband: float,
                     was_high: bool, was_low: bool) -> str:
    """습도 전용 밴드 판정(이슈 #33) — ON은 온도와 동일하게 밴드 밖 이탈 시지만, OFF는
    밴드 경계가 아니라 **밴드 중앙(mid) 근접 시**다(P-제어 목표가 중앙이므로 OFF 기준도
    중앙에 맞춘다). 지속(was_high/was_low) 중에는 |value-mid|가 deadband를 넘어야(=
    아직 중앙에서 충분히 먼) "high"/"low" 유지, deadband 이내로 들어오면 "normal"(OFF)."""
    mid = (low + high) / 2
    if value > high:
        return "high"
    if value < low:
        return "low"
    if was_high and abs(value - mid) > deadband:
        return "high"
    if was_low and abs(value - mid) > deadband:
        return "low"
    return "normal"


def decide(reading: dict, setpoints, states: dict[str, DeviceState], date=None,
           hum_mode: str = "edge", temp_mode: str = "edge") -> list[ControlLog]:
    """센서값+설정 밴드+현재 장치 상태 → 상태 변화(states 갱신) + 발생한 ControlLog 목록.

    수동(auto=False) 장치는 결정 대상에서 제외(현재 상태 그대로 유지, 로그 없음).
    heater/cooling_fan 동시 ON 금지, humidifier/dehumidifier 동시 ON 금지 —
    충돌 시 이탈 폭이 큰 쪽 우선(온도·습도는 서로 다른 밴드라 실제로는 발생하지 않음).

    hum_mode(이슈 #33 리뷰 P1 픽스) — 습도 판정 방식 선택:
    - "edge"(기본, 하위호환) — 경계 히스테리시스(_band_state, 온도와 동일 패턴). 리플레이
      (시뮬레이션 탭, app/views/monitor.py._run_control_step)는 EFFECTS_DAILY(1일 고정
      ±5%p) 스텝이라, 중앙(mid) OFF 창을 큰 스텝으로 건너뛰면 반대쪽 밴드까지 관통하는
      회귀가 있어 반드시 이 모드를 써야 한다.
    - "center" — 밴드 중앙(mid) 근접 시 OFF(_hum_band_state). live(오늘 운영,
      control.live.simulate_control)는 P-제어(시간당 델타가 중앙을 향해 비례 수렴)라
      이 모드를 명시적으로 사용한다.

    temp_mode(이슈 #45, hum_mode와 대칭) — 온도 판정 방식 선택:
    - "edge"(기본, 하위호환) — 경계 히스테리시스(_band_state). 리플레이(시뮬레이션 탭,
      app/views/monitor.py._run_control_step)는 이 기본값을 그대로 사용(변경 없음).
    - "center" — 밴드 중앙(mid) 근접 시 OFF(_hum_band_state 재사용 — 이름은 습도지만
      로직이 low/high/mid 기준 범용이라 온도에도 그대로 쓴다. 회귀 최소화를 위해
      리네임하지 않음). live(control.live.simulate_control)는 온도 P-제어(이슈 #45)
      도입으로 이 모드를 명시적으로 사용한다.
    """
    temp = reading.get("온도내부_평균", (setpoints.temp_low + setpoints.temp_high) / 2)
    hum = reading.get("습도내부_평균", (setpoints.hum_low + setpoints.hum_high) / 2)
    logs: list[ControlLog] = []

    def cur_on(device: str) -> bool:
        return states[device].on

    temp_band_fn = _hum_band_state if temp_mode == "center" else _band_state
    temp_state = temp_band_fn(temp, setpoints.temp_low, setpoints.temp_high,
                               setpoints.temp_deadband, cur_on("cooling_fan"), cur_on("heater"))
    hum_band_fn = _hum_band_state if hum_mode == "center" else _band_state
    hum_state = hum_band_fn(hum, setpoints.hum_low, setpoints.hum_high,
                             setpoints.hum_deadband, cur_on("dehumidifier"), cur_on("humidifier"))

    want_cooling = temp_state == "high"
    want_heater = temp_state == "low"
    if want_cooling and want_heater:  # 이론상 low<high면 불가하지만 방어적으로 처리
        if (temp - setpoints.temp_high) >= (setpoints.temp_low - temp):
            want_heater = False
        else:
            want_cooling = False

    want_dehumidifier = hum_state == "high"
    want_humidifier = hum_state == "low"
    if want_dehumidifier and want_humidifier:  # 이론상 low<high면 불가하지만 방어적으로 처리
        if (hum - setpoints.hum_high) >= (setpoints.hum_low - hum):
            want_humidifier = False
        else:
            want_dehumidifier = False

    def _apply(device: str, want: bool, reason: str):
        state = states[device]
        if not state.auto:
            return
        if state.on == want:
            return
        state.on = want
        logs.append(ControlLog(date=str(date), device=device, action="ON" if want else "OFF",
                                reason=reason, mode="auto"))

    _apply("cooling_fan", want_cooling,
           f"온도 상한 초과({temp:.1f}℃>{setpoints.temp_high:.1f}℃)" if want_cooling
           else "밴드 정상 범위 복귀")
    _apply("heater", want_heater,
           f"온도 하한 미달({temp:.1f}℃<{setpoints.temp_low:.1f}℃)" if want_heater
           else "밴드 정상 범위 복귀")
    _apply("dehumidifier", want_dehumidifier,
           f"습도 상한 초과({hum:.1f}%>{setpoints.hum_high:.1f}%)" if want_dehumidifier
           else "밴드 정상 범위 복귀")
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
            and states["cooling_fan"].on:
        candidates.append({"key": "control_limit:temp_high", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(냉방 풀가동에도 고온 지속)"})
    if all(t is not None and t < setpoints.temp_low for t in temps) and states["heater"].on:
        candidates.append({"key": "control_limit:temp_low", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(난방 풀가동에도 저온 지속)"})
    if all(h is not None and h > setpoints.hum_high for h in hums) and states["dehumidifier"].on:
        candidates.append({"key": "control_limit:hum_high", "level": "경고",
                            "reason": "제어 한계 초과 — 설비 점검 필요(제습기 풀가동에도 고습 지속)"})
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
