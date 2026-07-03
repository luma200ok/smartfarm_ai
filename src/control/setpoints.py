"""제어 설정 밴드 — 온도·습도 상/하한 + 히스테리시스 데드밴드(채터링 방지)."""
from dataclasses import dataclass


@dataclass
class Setpoints:
    """온도(℃)·습도(%) 밴드. ON은 밴드 밖에서, OFF는 데드밴드만큼 안쪽 복귀 시에만."""

    temp_low: float = 20.0
    temp_high: float = 25.0
    hum_low: float = 60.0
    hum_high: float = 85.0
    temp_deadband: float = 0.5
    hum_deadband: float = 2.0
