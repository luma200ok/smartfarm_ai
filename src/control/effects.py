"""장치 ON 효과 → VirtualSensor.inject() 피드백 델타(조정 가능한 상수로 분리)."""

# feature: delta(1일치). 히터=난방·쿨링팬=냉방·환기=온도소폭하강+제습·가습기=가습.
EFFECTS: dict[str, dict[str, float]] = {
    "heater": {"온도내부_평균": 1.5, "온도내부_최저": 1.5},
    "cooling_fan": {"온도내부_평균": -1.5, "온도내부_최저": -1.5},
    "vent": {"온도내부_평균": -0.5, "온도내부_최저": -0.5, "습도내부_평균": -5.0},
    "humidifier": {"습도내부_평균": 5.0},
}


def apply_effects(vs, states: dict, start, days: int = 1) -> None:
    """ON 상태인 장치들의 효과를 vs.inject()로 반영(다음 days일치, read-time overlay).

    tag="control"로 주입 — apply_scenario()의 "scenario" 태그 clear에 영향받지 않는다
    (이슈 #17 P1-1: 시나리오 재적용이 제어 효과까지 지우는 문제 방지)."""
    for device, state in states.items():
        if not state.on:
            continue
        for feature, delta in EFFECTS.get(device, {}).items():
            vs.inject(feature, start, days, delta, tag="control")
