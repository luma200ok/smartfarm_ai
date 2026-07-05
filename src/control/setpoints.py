"""제어 설정 밴드 — 온도·습도 상/하한 + 히스테리시스 데드밴드(채터링 방지)."""
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
    hum_high: float = 80.0  # 이슈 #51 — 중앙 유지형 전환에 맞춰 85→80(중앙 70%)
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
    """원자적 쓰기: tmp 파일에 쓴 뒤 os.replace로 교체.

    서버 권한·디스크 문제로 쓰기 실패 시 예외를 전파하지 않고 조용히 반환한다
    (세션 값은 이미 반영돼 있으므로 UI는 계속 동작 — 이슈 #21 P2-1).
    """
    from control import state_io

    target = Path(path) if path is not None else SETPOINTS_PATH
    data = {
        "temp_low": sp.temp_low,
        "temp_high": sp.temp_high,
        "hum_low": sp.hum_low,
        "hum_high": sp.hum_high,
    }
    state_io.save_json_atomic(target, data)


def load(path: "Path | None" = None) -> "Setpoints":
    """파일 없음·손상·키 누락 시 기본값 폴백(예외 전파 없음)."""
    from control import state_io

    target = Path(path) if path is not None else SETPOINTS_PATH
    default = Setpoints()
    raw = state_io.load_json(target)
    if not raw:
        return default
    try:
        sp = Setpoints(
            temp_low=float(raw["temp_low"]),
            temp_high=float(raw["temp_high"]),
            hum_low=float(raw["hum_low"]),
            hum_high=float(raw["hum_high"]),
        )
    except (KeyError, TypeError, ValueError):
        return default
    return _validate(sp)


_FIELDS = ("temp_low", "temp_high", "hum_low", "hum_high")


def save_changed(current: "Setpoints", prev: "Setpoints", path: "Path | None" = None) -> "Setpoints":
    """동시 세션 lost-update 방지(이슈 #21 P2-2).

    저장 직전 파일의 최신값을 베이스로 하고, `current`가 `prev` 대비 실제로
    바뀐 필드만 그 위에 병합해 저장한다. 다른 세션이 먼저 저장해둔 값(베이스에는
    있지만 이번 세션에서 안 건드린 필드)은 보존된다. 병합 결과를 반환하므로
    호출측이 세션 상태에도 반영할 수 있다.
    """
    latest = load(path)
    for field in _FIELDS:
        if getattr(current, field) != getattr(prev, field):
            setattr(latest, field, getattr(current, field))
    save(latest, path)
    return latest
