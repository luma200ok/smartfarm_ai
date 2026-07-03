"""장치 4종(vent·humidifier·cooling_fan·heater) 상태 컨테이너 + 제어 로그 항목."""
from dataclasses import dataclass

DEVICES: tuple[str, ...] = ("vent", "humidifier", "cooling_fan", "heater")

DEVICE_LABEL_KR = {
    "vent": "환기",
    "humidifier": "가습기",
    "cooling_fan": "쿨링팬",
    "heater": "히터",
}


@dataclass
class DeviceState:
    """장치 1개의 현재 상태 — on(ON/OFF)·auto(자동/수동 모드)."""

    on: bool = False
    auto: bool = True


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
