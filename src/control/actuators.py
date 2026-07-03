"""장치 4종(dehumidifier·humidifier·cooling_fan·heater) 상태 컨테이너 + 제어 로그 항목.

이슈 #27 — 환기(vent)를 제거하고 제습기(dehumidifier)를 신설해 온도(히터/쿨링팬)·습도
(가습기/제습기) 각 2종씩 대칭 구조로 재편했다."""
from dataclasses import dataclass

DEVICES: tuple[str, ...] = ("dehumidifier", "humidifier", "cooling_fan", "heater")

DEVICE_LABEL_KR = {
    "dehumidifier": "제습기",
    "humidifier": "가습기",
    "cooling_fan": "쿨링팬",
    "heater": "히터",
}


@dataclass
class DeviceState:
    """장치 1개의 현재 상태 — on(ON/OFF)·auto(자동/수동 모드).

    cause: 현재 ON 상태를 유발한 판정 출처("temp"|"hum"|None) — 이슈 #27로 습도 장치가
    dehumidifier/humidifier 전용이 되면서 더 이상 두 규칙을 공유하지 않지만, ControlLog와
    동일한 필드 형태를 유지하기 위해 남겨둔다."""

    on: bool = False
    auto: bool = True
    cause: str | None = None


@dataclass
class ControlLog:
    """제어 로그 1건 — date/device/action(ON·OFF)/reason/mode(auto·manual)."""

    date: str
    device: str
    action: str
    reason: str
    mode: str = "auto"


def default_states() -> dict[str, DeviceState]:
    """장치 4종 모두 OFF·자동 모드로 초기화된 상태 컨테이너."""
    return {d: DeviceState() for d in DEVICES}
