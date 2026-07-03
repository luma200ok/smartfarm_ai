"""제어 설정 밴드 — 온도·습도 상/하한 + 히스테리시스 데드밴드(채터링 방지)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETPOINTS_PATH = ROOT / "data" / "control_setpoints.json"


@dataclass
class Setpoints:
    """온도(℃)·습도(%) 밴드. ON은 밴드 밖에서, OFF는 데드밴드만큼 안쪽 복귀 시에만."""

    temp_low: float = 20.0
    temp_high: float = 25.0
    hum_low: float = 60.0
    hum_high: float = 85.0
    temp_deadband: float = 0.5
    hum_deadband: float = 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _validate(sp: "Setpoints") -> "Setpoints":
    """범위 클램프 + low<high 검증. 위반 시 기본값 폴백."""
    default = Setpoints()
    sp.temp_low = _clamp(sp.temp_low, 0.0, 40.0)
    sp.temp_high = _clamp(sp.temp_high, 0.0, 40.0)
    sp.hum_low = _clamp(sp.hum_low, 0.0, 100.0)
    sp.hum_high = _clamp(sp.hum_high, 0.0, 100.0)
    if sp.temp_low >= sp.temp_high:
        sp.temp_low, sp.temp_high = default.temp_low, default.temp_high
    if sp.hum_low >= sp.hum_high:
        sp.hum_low, sp.hum_high = default.hum_low, default.hum_high
    # deadband는 파일 값 무시 — 슬라이더 미노출이라 항상 기본값 유지
    sp.temp_deadband = default.temp_deadband
    sp.hum_deadband = default.hum_deadband
    return sp


def save(sp: "Setpoints", path: "Path | None" = None) -> None:
    """원자적 쓰기: tmp 파일에 쓴 뒤 os.replace로 교체."""
    target = Path(path) if path is not None else SETPOINTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    data = {
        "temp_low": sp.temp_low,
        "temp_high": sp.temp_high,
        "hum_low": sp.hum_low,
        "hum_high": sp.hum_high,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, target)


def load(path: "Path | None" = None) -> "Setpoints":
    """파일 없음·손상·키 누락 시 기본값 폴백(예외 전파 없음)."""
    target = Path(path) if path is not None else SETPOINTS_PATH
    default = Setpoints()
    if not target.exists():
        return default
    try:
        with open(target, "r", encoding="utf-8") as f:
            raw = json.load(f)
        sp = Setpoints(
            temp_low=float(raw["temp_low"]),
            temp_high=float(raw["temp_high"]),
            hum_low=float(raw["hum_low"]),
            hum_high=float(raw["hum_high"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return default
    return _validate(sp)
