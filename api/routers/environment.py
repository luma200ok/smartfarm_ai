"""GET /api/environment/today — 오늘 운영(관제) 요약 조회(smartfarm_ai#66, "ai-server 무변경
원칙" 해제 결정 2026-08-20, 서비스 PRD §11-5).

**새 계산 로직을 작성하지 않는다** — `control/live.py`의 기존 상태/스냅샷 파일만 조회한다.
서비스가 60s 캐시로 반복 호출하는 엔드포인트라, 타임라인 전체를 재계산하는
`assemble_today_timeline()`(KMA 다일치 예보 파싱 + `simulate_control` 재생) 대신:
  - 실내값: 이미 기록된 오늘 스냅샷(`load_today_snapshots`, 상태 파일 1회 읽기)에서
    "지금"에 가장 가까운 과거 시간 항목의 ctrl_temp/ctrl_hum(제어 후 값)만 꺼내 쓴다.
  - 실외값: KMA 3일 예보 전체(`today_outdoor`) 대신 실황 단건 조회(`weather.get_current`)만
    호출한다 — 오늘 운영 알림(run_notify)이 쓰는 예보 파싱 경로와 분리된 가벼운 경로.

demo=true 고정 — 1차 배포는 농장↔센서 매핑 이전(전 농장 공용 데모 온실 데이터), 응답에
명시해 서비스가 구분 표시할 수 있게 한다(후속: farm_id별 매핑, 이 라우터 범위 밖).

가용성 원칙(레포 기존 관용, health.py의 "Ollama 오프라인이어도 200"과 동일) — KMA·상태 파일이
불가해도 5xx를 내지 않고 항상 200 + 가용 필드만 채워 반환한다. 불가 사유·데이터 결측/오래됨은
alerts 배열의 안내 문자열로만 전달한다(런타임 로직에 영향 없음).
"""
from datetime import datetime

from control.actuators import DEVICES
from control.live import load_today_snapshots
from fastapi import APIRouter
from llm import weather

from ..schemas import DeviceStatus, EnvironmentTodayResponse, IndoorReading, OutdoorReading

router = APIRouter(tags=["environment"])


def _latest_snapshot(snapshots: dict, now_hour: int) -> "tuple[dict | None, int | None]":
    """오늘 스냅샷 중 "지금"(now_hour) 이하 시간대에서 가장 최근 항목 — (snapshot, hour).
    아직 하나도 기록되지 않았으면 (None, None)."""
    hours = sorted(int(h) for h in snapshots if int(h) <= now_hour)
    if not hours:
        return None, None
    hour = hours[-1]
    return snapshots[str(hour)], hour


def _recorded_at(snap: dict, fallback: datetime) -> datetime:
    """스냅샷의 recorded_at(ISO 문자열) → datetime. 파싱 실패/결측 시 fallback(서버 현재 시각)."""
    raw = snap.get("recorded_at")
    if not raw:
        return fallback
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return fallback


@router.get("/environment/today", response_model=EnvironmentTodayResponse)
def get_environment_today() -> EnvironmentTodayResponse:
    now = datetime.now()
    alerts: list[str] = []

    current = weather.get_current()
    if current.get("unavailable"):
        alerts.append(f"외기 실황 조회 실패: {current.get('reason', 'KMA 불가')}")
        outdoor = OutdoorReading()
    else:
        outdoor = OutdoorReading(temp=current.get("temp"), humidity=current.get("humidity"))

    snapshots = load_today_snapshots(now.date())
    snap, snap_hour = _latest_snapshot(snapshots, now.hour)

    if snap is None:
        alerts.append("오늘 기록된 운영 데이터 없음")
        indoor = IndoorReading()
        devices: list[DeviceStatus] = []
        updated_at = now
    else:
        indoor = IndoorReading(temp=snap.get("ctrl_temp"), humidity=snap.get("ctrl_hum"))
        devices_on = set(snap.get("devices_on") or [])
        devices = [DeviceStatus(name=d, on=d in devices_on) for d in DEVICES]
        updated_at = _recorded_at(snap, fallback=now)
        if snap_hour is not None and snap_hour < now.hour:
            alerts.append(f"최근 기록 {snap_hour}시 — 이후 갱신 없음")

    return EnvironmentTodayResponse(
        demo=True, updated_at=updated_at, outdoor=outdoor, indoor=indoor,
        devices=devices, alerts=alerts,
    )
