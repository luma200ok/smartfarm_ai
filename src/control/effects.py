"""장치 ON 효과 → VirtualSensor.inject() 피드백 델타(조정 가능한 상수로 분리).

이슈 #27 — 오늘 운영(control/live.py, 시간 단위 시뮬레이션)은 EFFECTS_HOURLY(시간당
상수)를 직접 쓰고, 시뮬레이션 탭(리플레이, 1일 틱)은 기존 동작을 유지하기 위해
EFFECTS_DAILY(1일치 풀가동 델타)를 그대로 쓴다 — apply_effects()는 항상 DAILY."""

# feature: delta(1시간당). heater=난방·cooling_fan=냉방·humidifier=가습·dehumidifier=제습.
# 효과를 드라마틱하게(사용자 확정) — 온도 ±2.0℃/h, 습도 ±8.0%p/h.
EFFECTS_HOURLY: dict[str, dict[str, float]] = {
    "heater": {"온도내부_평균": 2.0, "온도내부_최저": 2.0},
    "cooling_fan": {"온도내부_평균": -2.0, "온도내부_최저": -2.0},
    "humidifier": {"습도내부_평균": 8.0},
    "dehumidifier": {"습도내부_평균": -8.0},
}

# feature: delta(1일치). 히터=난방·쿨링팬=냉방·제습기=제습·가습기=가습.
# (이슈 #27: vent 제거 — 온도 효과 없이 습도만 제습기로 이관, 값은 기존 vent 습도효과 유지)
EFFECTS_DAILY: dict[str, dict[str, float]] = {
    "heater": {"온도내부_평균": 1.5, "온도내부_최저": 1.5},
    "cooling_fan": {"온도내부_평균": -1.5, "온도내부_최저": -1.5},
    "dehumidifier": {"습도내부_평균": -5.0},
    "humidifier": {"습도내부_평균": 5.0},
}

# 하위호환 별칭(과거 EFFECTS 참조부 대비) — apply_effects()는 DAILY를 사용.
EFFECTS = EFFECTS_DAILY


def apply_effects(vs, states: dict, start, days: int = 1) -> None:
    """ON 상태인 장치들의 효과를 vs.inject()로 반영(다음 days일치, read-time overlay).

    tag="control"로 주입 — apply_scenario()의 "scenario" 태그 clear에 영향받지 않는다
    (이슈 #17 P1-1: 시나리오 재적용이 제어 효과까지 지우는 문제 방지). 시뮬레이션 탭(리플레이)
    전용이라 EFFECTS_DAILY를 사용(이슈 #27 — 리플레이 로직 변경 없음 원칙)."""
    for device, state in states.items():
        if not state.on:
            continue
        for feature, delta in EFFECTS_DAILY.get(device, {}).items():
            vs.inject(feature, start, days, delta, tag="control")
