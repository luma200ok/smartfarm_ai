"""오늘 운영 모드(이슈 #23) — KMA 외기 + 기대값 모델로 오늘 시간대별
"제어 전 내부 기준선"과 "장치 효과가 반영된 제어 후 내부값"을 계산한다.

규칙 기반(controller.decide()·effects.EFFECTS_HOURLY 재사용), LLM 호출 없음.
run_notify()는 서버 타이머(1시간 간격, systemd timer)에서 호출되는 진입점 —
장치 ON/OFF 전환·긴급·이상만 디스코드로 발송(상태 파일 dedup, 세션이 아니라
파일 기반이라 앱·타이머 프로세스가 상태를 공유한다).
"""
import argparse
import logging
from datetime import date as _date
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
ROOT = _SRC.parent

_log = logging.getLogger(__name__)

STATE_PATH = ROOT / "data" / "control_live_state.json"
HISTORY_PATH = ROOT / "data" / "control_history.json"
HISTORY_MAX_DAYS = 30  # 아카이브 보존 기간(archive_snapshots)

# 물리 클램프(이슈 #27) — 효과가 시간당 상수(EFFECTS_HOURLY)로 커진 만큼, 극단적인 연속
# ON에도 ctrl_temp/ctrl_hum이 비현실적으로 발산하지 않도록 base 대비 편차를 제한한다.
CTRL_TEMP_BAND = 10.0  # base_temp ± 이 범위(℃) 안에서만 ctrl_temp 허용
CTRL_HUM_MIN = 0.0
CTRL_HUM_MAX = 100.0

# 실내 습도 기준선 근사 — 외기습도 + 온실 보습 보정(상수, 실측 아닌 근사치).
INDOOR_HUMIDITY_OFFSET = 10.0

# 습도 P-제어(이슈 #33) — 장치(제습기/가습기) ON 중 시간당 델타 = clamp(HUM_P_GAIN ×
# (hum_mid - ctrl_hum), -HUM_P_MAX_DELTA, +HUM_P_MAX_DELTA). hum_mid = 밴드 중앙
# ((hum_low+hum_high)/2). 기본 밴드(60~85%, mid=72.5%) 기준 오차 13.3%p 이상이면 캡
# (±8.0%p/h, 기존 EFFECTS_HOURLY 습도 값과 동일 — "최대 출력"으로 의미 재정의)에
# 도달하는 수준으로 튜닝. 중앙에 가까워질수록 델타가 비례해 줄어 진동 없이 수렴한다.
HUM_P_GAIN = 0.6
HUM_P_MAX_DELTA = 8.0

# 일사량 시간별 데이터가 없어 주간(07~18시)만 "낮"으로 근사(학습 데이터 doy 계절 평균 사용),
# 야간은 0으로 둔다.
DAYTIME_HOURS = range(7, 19)

# KMA 조회가 연속 이만큼 실패하면 이상 알림(일시적 429 등 노이즈 억제)
_FAIL_ALERT_THRESHOLD = 2


def today_outdoor(today: "_date | None" = None, now: "datetime | None" = None) -> "list[dict] | None":
    """오늘(today) 시간대별 외기 {hour, temp, humidity} — 현재 시각 항목은 실황으로 대체.

    KMA unavailable(키 미설정·호출 실패·오늘 데이터 없음)이면 None(우아한 저하).
    """
    from llm import weather

    now = now or datetime.now()
    today = today or now.date()

    fcst = weather.get_forecast_3d()
    if fcst.get("unavailable"):
        return None

    date_str = today.strftime("%Y%m%d")
    hourly = [h for h in fcst.get("hourly", []) if h.get("date") == date_str]
    if not hourly:
        return None

    out = []
    for h in hourly:
        raw_time = h.get("time")
        if not raw_time or len(str(raw_time)) < 2:
            continue
        try:
            hour = int(str(raw_time)[:2])
        except ValueError:
            continue
        out.append({"hour": hour, "temp": h.get("temp"), "humidity": h.get("humidity")})
    if not out:
        return None
    out.sort(key=lambda x: x["hour"])

    current = weather.get_current()
    if not current.get("unavailable") and current.get("temp") is not None:
        cur_hour = now.hour
        for item in out:
            if item["hour"] == cur_hour:
                item["temp"] = current["temp"]
                if current.get("humidity") is not None:
                    item["humidity"] = current["humidity"]
                break
        else:
            out.append({"hour": cur_hour, "temp": current["temp"], "humidity": current.get("humidity")})
            out.sort(key=lambda x: x["hour"])

    return out


def indoor_baseline(outdoor: "list[dict]", date=None) -> "list[dict]":
    """시간별 외기 → expect.predict()로 시간별 내부 기준선(제어 전) 추정 — 근사치.

    일사량은 시간별 실측이 없어 주간(07~18시)=학습 데이터 doy 계절 평균
    (model["doy_solar_climatology"]), 야간=0으로 근사해 predict()에 넣는다.
    내부 습도 기준선(이슈 #37)은 모델 payload에 "습도" 타깃이 있으면 predict() 결과를
    그대로 사용하고, 없으면(구버전 pkl) 기존 폴백 — 외기 습도 + INDOOR_HUMIDITY_OFFSET
    (온실 보습 보정, 상수)로 단순 근사한다 — 어느 쪽이든 정밀 물리 모델이 아니라
    대시보드용 근사치임에 유의.
    기대값 모델(pkl) 미배포 시 base_temp는 외기 온도 그대로 폴백한다.
    """
    from llm import expect as expect_mod

    model = expect_mod.load_model()
    solar_clim = (model or {}).get("doy_solar_climatology", {}) or {}
    d = date or datetime.now().date()
    doy = expect_mod.doy(d)
    solar = solar_clim.get(doy, 0.0)

    baseline = []
    for item in outdoor:
        hour = item["hour"]
        temp = item.get("temp")
        humidity = item.get("humidity")
        irradiance = solar if hour in DAYTIME_HOURS else 0.0

        exp = None
        if temp is not None:
            exp = expect_mod.predict(model, {"온도외부_평균": temp, "일사량_평균": irradiance}, d)
        base_temp = exp["평균"] if exp else temp

        base_hum = None
        if exp and "습도" in exp:
            base_hum = max(0.0, min(exp["습도"], 100.0))
        elif humidity is not None:
            base_hum = min(humidity + INDOOR_HUMIDITY_OFFSET, 100.0)

        baseline.append({
            "hour": hour, "out_temp": temp, "out_hum": humidity,
            "base_temp": base_temp, "base_hum": base_hum,
        })
    return baseline


def _hum_pcontrol_delta(devices_on: "list[str]", ctrl_hum: float, setpoints) -> float:
    """습도 P-제어 델타(이슈 #33) — 제습기/가습기 ON 중에만 hum_mid를 향해 비례 이동.
    둘 다 OFF면 0.0(장치 효과 없음). decide()가 dehumidifier/humidifier 동시 ON을
    금지하므로 두 장치가 동시에 이 함수에 들어올 일은 없다."""
    if "dehumidifier" not in devices_on and "humidifier" not in devices_on:
        return 0.0
    hum_mid = (setpoints.hum_low + setpoints.hum_high) / 2
    raw = HUM_P_GAIN * (hum_mid - ctrl_hum)
    return max(-HUM_P_MAX_DELTA, min(HUM_P_MAX_DELTA, raw))


def _clamp_into_band(value, low: float, high: float, deadband: float):
    """밴드 안쪽(low+deadband ~ high-deadband)으로 클램프 — 자정 연속성 폴백(이슈 #35)에서
    "이전에도 제어 운영 중이었다" 가정으로 0시 시작값을 밴드 밖 기준선 대신 안쪽에서
    출발시킬 때 사용. value가 None이면 그대로 None."""
    if value is None:
        return None
    lo, hi = low + deadband, high - deadband
    if lo > hi:  # deadband가 밴드 폭보다 크면 퇴화 — 중앙값으로 폴백
        return (low + high) / 2
    return max(lo, min(hi, value))


def simulate_control(baseline: "list[dict]", setpoints, states, date=None,
                      initial_ctrl: "dict | None" = None,
                      fallback_clamp: bool = False,
                      seed_ctrl: "tuple[float | None, float | None] | None" = None) -> "list[dict]":
    """시간 순 재생 — controller.decide() 재사용, ON 장치 효과(시간 환산)를 다음 시간
    내부값에 반영해 "제어 후" 내부온·습도를 산출한다.

    자정 연속성(이슈 #35) — 매일 0시를 기준선에서 새로 시작하면 어제 제어로 밴드 안에
    있었던 값이 오늘 0시엔 밴드 밖(기준선 그대로)에서 출발하는 불연속이 생긴다(제어
    상시 운영 전제와 모순). 이를 완화하기 위해:
    - `initial_ctrl`이 주어지고 그 "date"가 정확히 어제(전일 마지막 실행 값으로 유효)면,
      0시(baseline 첫 항목) 시작값을 기준선 대신 `initial_ctrl["temp"]`/`["hum"]`로
      이어받는다. 오늘 날짜는 인정하지 않는다 — run_notify는 하루에 여러 번 호출되는
      진입점이라 "오늘"까지 인정하면 호출마다 0시가 직전 호출의 "지금" 값으로 재시딩돼
      하루 안에서도 타임라인이 드리프트하며 dedup·멱등성이 깨진다.
    - `initial_ctrl`이 없거나 무효(그제 이전/None)하고 `fallback_clamp=True`면, "이전에도
      제어 운영 중이었다"는 가정 하에 시작값을 밴드 안쪽(low+deadband~high-deadband)으로
      클램프한다 — 기준선이 밴드 밖이어도 0시 ctrl은 밴드 안에서 시작.
    - 둘 다 아니면(기본값, `fallback_clamp=False`) 기존 거동(기준선 그대로 시작) 유지 —
      순수 시뮬레이션/리플레이·기존 테스트 무회귀용 기본값.

    `seed_ctrl`(이슈 #40, 오늘 스냅샷 경계 합성 전용) — `(ctrl_temp, ctrl_hum)` 튜플이
    주어지면 위 initial_ctrl 유효성 검증(어제-전용)을 거치지 않고 0시(baseline 첫 항목)
    시작값으로 그대로 사용한다 — 과거 스냅샷의 마지막 기록에서 미래 구간을 이어 합성할 때,
    baseline이 "오늘의 남은 시간대"만 담아 idx 0이 실제로는 자정이 아니기 때문. 기본값
    None이면 기존 initial_ctrl/fallback_clamp 경로가 그대로 동작(완전 무회귀). run_notify는
    seed_ctrl을 사용하지 않는다(위 initial_ctrl 불변식 유지).

    제어 관성(누적) 방식(이슈 #27 리뷰 P2-1/2-2 픽스) — 매 시간 baseline으로 재기준하지
    않고, 이전 제어 결과(ctrl_prev)에 외기 변화분(base_next-base_prev)과 장치 효과
    (delta_hourly)를 더해 다음 값을 구한다. 가변 외기에서 매 시간 baseline으로 재기준하면
    ①채터링(장치 ON→다음 시간 baseline이 낮아 즉시 OFF 판정) ②한 스텝에 밴드 반대쪽을
    관통(냉방으로 temp_low 아래까지 내려가 heater ON — 교대 진동)하는 문제가 있었다.

    한 스텝 관통 방지: 냉방/난방/제습/가습 ON 중에는 그 장치가 미는 방향의 반대쪽 데드밴드
    경계(예: 냉방 중이면 temp_low+deadband)를 넘어가지 않도록 클램프한다 — 다음 스텝에서
    반대 장치가 즉시 켜지는 교대 진동을 막는다. 기존 물리 클램프(base±CTRL_TEMP_BAND℃·
    습도 0~100%)는 그대로 유지(비현실적 발산 방지).

    반환: [{hour, out_temp, base_temp, ctrl_temp, out_hum, base_hum, ctrl_hum, devices_on, events}]
    (events = 그 시간대에 발생한 ControlLog 목록).
    """
    from control import controller
    from control.effects import EFFECTS_HOURLY

    date = date or datetime.now().date()

    # 유효성: last_ctrl의 date는 "어제"만 인정한다 — 오늘 날짜(같은 날 재실행)까지 인정하면
    # 매 호출마다 0시를 직전 호출의 "지금" 값으로 재시딩해 하루 안에서도 타임라인이 호출마다
    # 갱신되며 드리프트한다(run_notify는 같은 날 여러 번 호출되는 진입점이라 dedup·멱등성이
    # 깨짐). 자정 연속성은 "어제 마지막 값 → 오늘 0시" 전환에서만 의미가 있으므로 어제로 제한.
    seed_temp = seed_hum = None
    if initial_ctrl and isinstance(date, _date):
        from datetime import timedelta
        yesterday_str = (date - timedelta(days=1)).isoformat()
        if initial_ctrl.get("date") == yesterday_str:
            seed_temp = initial_ctrl.get("temp")
            seed_hum = initial_ctrl.get("hum")

    if seed_ctrl is not None:  # 명시 시드가 최우선(경계 합성 전용, 위 유효성 검증 우회)
        seed_temp, seed_hum = seed_ctrl

    timeline = []
    ctrl_temp = None
    ctrl_hum = None

    for idx, item in enumerate(baseline):
        if ctrl_temp is None:
            if seed_temp is not None:
                ctrl_temp = seed_temp
            elif fallback_clamp:
                ctrl_temp = _clamp_into_band(item["base_temp"], setpoints.temp_low,
                                              setpoints.temp_high, setpoints.temp_deadband)
            else:
                ctrl_temp = item["base_temp"]
        if ctrl_hum is None:
            if seed_hum is not None:
                ctrl_hum = seed_hum
            elif fallback_clamp:
                ctrl_hum = _clamp_into_band(item["base_hum"], setpoints.hum_low,
                                             setpoints.hum_high, setpoints.hum_deadband)
            else:
                ctrl_hum = item["base_hum"]

        reading = {"온도내부_평균": ctrl_temp, "습도내부_평균": ctrl_hum}
        # hum_mode="center" — P-제어(델타가 hum_mid로 비례 수렴)에 맞춰 OFF 판정도 중앙
        # 근접 기준을 명시(리뷰 P1 픽스, 이슈 #33) — 리플레이(app/views/monitor.py)는
        # decide() 기본값(edge, 경계 히스테리시스)을 그대로 사용.
        logs = controller.decide(reading, setpoints, states, date=f"{date} {item['hour']:02d}:00",
                                  hum_mode="center")
        devices_on = [d for d in states if states[d].on]

        timeline.append({
            "hour": item["hour"], "out_temp": item["out_temp"], "base_temp": item["base_temp"],
            "ctrl_temp": ctrl_temp, "out_hum": item.get("out_hum"),
            "base_hum": item["base_hum"], "ctrl_hum": ctrl_hum,
            "devices_on": devices_on, "events": logs,
        })

        delta_temp = sum(EFFECTS_HOURLY.get(d, {}).get("온도내부_평균", 0.0) for d in devices_on)
        delta_hum = _hum_pcontrol_delta(devices_on, ctrl_hum, setpoints)

        next_base_temp = baseline[idx + 1]["base_temp"] if idx + 1 < len(baseline) else None
        next_base_hum = baseline[idx + 1]["base_hum"] if idx + 1 < len(baseline) else None

        if next_base_temp is not None:
            base_drift_temp = next_base_temp - item["base_temp"]
            candidate = ctrl_temp + base_drift_temp + delta_temp
            # 물리 클램프(다음 base_temp ± CTRL_TEMP_BAND, 과도 발산 방지)를 먼저 적용하고,
            # 관통 방지(데드밴드 경계)는 마지막에 적용해 항상 우선한다 — 외기가 급변해
            # 물리 클램프가 관통 방지 경계보다 더 좁게 잡히는 극단적 케이스에서도 "냉방/난방
            # 중엔 반대쪽 데드밴드를 넘지 않는다"는 안전 불변식이 깨지지 않도록 한다.
            candidate = max(next_base_temp - CTRL_TEMP_BAND, min(next_base_temp + CTRL_TEMP_BAND, candidate))
            if "cooling_fan" in devices_on:  # 냉방 중 — temp_low 데드밴드 경계 아래로 관통 금지
                candidate = max(candidate, setpoints.temp_low + setpoints.temp_deadband)
            if "heater" in devices_on:  # 난방 중 — temp_high 데드밴드 경계 위로 관통 금지
                candidate = min(candidate, setpoints.temp_high - setpoints.temp_deadband)
            ctrl_temp = candidate
        else:
            ctrl_temp = None

        if next_base_hum is not None:
            base_drift_hum = next_base_hum - item["base_hum"]
            candidate_hum = ctrl_hum + base_drift_hum + delta_hum
            # 물리 클램프(0~100%)는 그대로 유지. 관통 방지는 P-제어 도입(이슈 #33)으로
            # 델타 자체가 hum_mid를 향해 비례 수렴해 밴드 중앙을 크게 넘기지 않지만, 외기
            # 급변(base_drift_hum)이 한 스텝에 더해질 수 있어 중앙(hum_mid) 기준 데드밴드
            # 경계는 안전망으로 유지한다(기존 hum_low/high 경계 대신 hum_mid 경계로 조정 —
            # OFF 판정 기준이 밴드 경계가 아니라 중앙 근접으로 바뀌었으므로 일치시킴).
            candidate_hum = max(CTRL_HUM_MIN, min(CTRL_HUM_MAX, candidate_hum))
            hum_mid = (setpoints.hum_low + setpoints.hum_high) / 2
            if "dehumidifier" in devices_on:  # 제습 중 — 중앙 데드밴드 아래로 관통 금지
                candidate_hum = max(candidate_hum, hum_mid - setpoints.hum_deadband)
            if "humidifier" in devices_on:  # 가습 중 — 중앙 데드밴드 위로 관통 금지
                candidate_hum = min(candidate_hum, hum_mid + setpoints.hum_deadband)
            ctrl_hum = candidate_hum
        else:
            ctrl_hum = None

    return timeline


def emergency_hours(timeline: "list[dict]", setpoints) -> "list[dict]":
    """풀가동(관련 장치 전부 ON)에도 제어 후 값이 밴드 밖인 시간대 → [{hour, reason}, ...]."""
    out = []
    for item in timeline:
        temp = item["ctrl_temp"]
        hum = item["ctrl_hum"]
        devices_on = set(item["devices_on"])
        if temp is not None:
            if temp > setpoints.temp_high and "cooling_fan" in devices_on:
                out.append({"hour": item["hour"], "kind": "temp_high",
                            "reason": f"제어 한계 초과 — 냉방 풀가동에도 고온 지속({temp:.1f}℃)"})
            elif temp < setpoints.temp_low and "heater" in devices_on:
                out.append({"hour": item["hour"], "kind": "temp_low",
                            "reason": f"제어 한계 초과 — 난방 풀가동에도 저온 지속({temp:.1f}℃)"})
        if hum is not None:
            if hum > setpoints.hum_high and "dehumidifier" in devices_on:
                out.append({"hour": item["hour"], "kind": "hum_high",
                            "reason": f"제어 한계 초과 — 제습기 풀가동에도 고습 지속({hum:.1f}%)"})
            elif hum < setpoints.hum_low and "humidifier" in devices_on:
                out.append({"hour": item["hour"], "kind": "hum_low",
                            "reason": f"제어 한계 초과 — 가습 풀가동에도 저습 지속({hum:.1f}%)"})
    return out


def _emergency_key(item: dict) -> str:
    """긴급 dedup 키 — hour만으로 묶으면 같은 시간대에 다른 사유(예: 고온→고습)가 잇달아
    발생해도 먼저 잡힌 사유가 나머지를 영구 억제한다(P2-2). kind(사유 요약, 안정적 카테고리)를
    함께 넣어 구분한다 — reason 원문은 온도·습도 수치가 매번 미세하게 달라 그대로 키로 쓰면
    dedup이 사실상 무력화되므로 사용하지 않는다."""
    return f"{item['hour']}:{item['kind']}"


# ── 오늘 시간별 스냅샷(이슈 #40) — KMA가 최신 발표분 이후 시간대만 제공해 저녁이 될수록
# 과거 시간대를 잃는 문제를 상태 파일에 시간별 기록을 누적해 해결한다.
# 과거=기록된 스냅샷 그대로, 현재·미래=예보 합성(assemble_today_timeline) ────────────

def _serialize_events(logs: list) -> "list[dict]":
    """timeline 행의 events(ControlLog 목록 또는 이미 dict인 스냅샷 복원값)를 JSON
    직렬화 가능한 dict 목록으로 통일한다."""
    out = []
    for log in logs:
        if isinstance(log, dict):
            out.append(log)
        else:
            out.append({"device": log.device, "action": log.action,
                        "reason": log.reason, "mode": getattr(log, "mode", "auto")})
    return out


def record_snapshot(state: dict, item: dict, source: str = "sim") -> bool:
    """timeline 행(item)을 state["snapshots"][str(hour)]에 기록한다 — first-write-wins
    (이미 기록돼 있으면 덮지 않고 False 반환, 과거 기록의 불변성을 보장). state는 in-place로
    수정된다(호출부가 저장 여부를 결정)."""
    hour = item["hour"]
    snapshots = state.setdefault("snapshots", {})
    key = str(hour)
    if key in snapshots:
        return False
    snapshots[key] = {
        "out_temp": item.get("out_temp"), "out_hum": item.get("out_hum"),
        "base_temp": item.get("base_temp"), "base_hum": item.get("base_hum"),
        "ctrl_temp": item.get("ctrl_temp"), "ctrl_hum": item.get("ctrl_hum"),
        "devices_on": list(item.get("devices_on") or []),
        "events": _serialize_events(item.get("events") or []),
        "source": source,
        "recorded_at": datetime.now().isoformat(),
    }
    state["version"] = 2
    return True


def load_today_snapshots(today: "_date | None" = None) -> dict:
    """오늘 날짜의 스냅샷 dict({"14": {...}, ...}) — 상태 파일 date가 today와 다르면
    빈 dict. 호출부(assemble_today_timeline 등)가 상태 파일 내부 포맷에 직접 결합되지
    않도록 하는 공개 API(load_last_ctrl과 동일한 취지)."""
    today = today or datetime.now().date()
    state = _load_state()
    if state.get("date") != today.isoformat():
        return {}
    return state.get("snapshots", {})


def _snapshot_to_row(hour: int, snap: dict) -> dict:
    """저장된 스냅샷 1건을 simulate_control() 타임라인 행과 동일한 포맷으로 복원한다."""
    return {
        "hour": hour,
        "out_temp": snap.get("out_temp"), "base_temp": snap.get("base_temp"),
        "ctrl_temp": snap.get("ctrl_temp"), "out_hum": snap.get("out_hum"),
        "base_hum": snap.get("base_hum"), "ctrl_hum": snap.get("ctrl_hum"),
        "devices_on": list(snap.get("devices_on") or []),
        "events": list(snap.get("events") or []),
    }


def archive_snapshots(prev_state: dict) -> None:
    """날짜가 바뀔 때 직전 날짜의 스냅샷을 data/control_history.json에 이관한다
    ({prev_date: snapshots} 병합) — 최근 HISTORY_MAX_DAYS일만 보존. prev_state에
    date·snapshots가 없으면 아무 것도 하지 않는다(예외 전파 없음)."""
    from control import state_io

    prev_date = prev_state.get("date")
    snapshots = prev_state.get("snapshots")
    if not prev_date or not snapshots:
        return

    history = state_io.load_json(HISTORY_PATH)
    history[prev_date] = snapshots
    if len(history) > HISTORY_MAX_DAYS:
        keep_keys = sorted(history.keys())[-HISTORY_MAX_DAYS:]
        history = {k: history[k] for k in keep_keys}
    state_io.save_json_atomic(HISTORY_PATH, history)


def assemble_today_timeline(outdoor: "list[dict] | None", setpoints, states, today: "_date",
                             now_hour: int) -> "list[dict]":
    """오늘 타임라인 조립(이슈 #40) — 과거(hour < now_hour)는 기록된 스냅샷으로 복원하고,
    현재·미래(hour >= now_hour)는 그 구간 outdoor로 새로 합성한다. 전달받은 states는
    복사본으로만 사용해 호출부의 원본을 변형하지 않는다.

    outdoor=None(KMA 실패)이면 과거 스냅샷 행만 반환한다(빈 리스트일 수 있음).
    """
    from copy import deepcopy

    snapshots = load_today_snapshots(today)
    past_hours = sorted(int(h) for h in snapshots if int(h) < now_hour)
    past_rows = [_snapshot_to_row(h, snapshots[str(h)]) for h in past_hours]

    if outdoor is None:
        return past_rows

    # 기록 없는 과거 시간 — outdoor(예보)에 남아 있으면 시뮬값으로 폴백, 없으면 결측 생략.
    recorded_hours = set(int(h) for h in snapshots)
    missing_past_outdoor = [o for o in outdoor
                             if o["hour"] < now_hour and o["hour"] not in recorded_hours]
    if missing_past_outdoor:
        missing_baseline = indoor_baseline(missing_past_outdoor, date=today)
        missing_rows = simulate_control(missing_baseline, setpoints, deepcopy(states),
                                         date=today, fallback_clamp=True)
        past_rows.extend(missing_rows)
        past_rows.sort(key=lambda r: r["hour"])

    future_outdoor = [o for o in outdoor if o["hour"] >= now_hour]
    if not future_outdoor:
        return past_rows

    future_baseline = indoor_baseline(future_outdoor, date=today)
    seed_ctrl = None
    if past_hours:
        last_snap = snapshots[str(past_hours[-1])]
        seed_ctrl = (last_snap.get("ctrl_temp"), last_snap.get("ctrl_hum"))

    if seed_ctrl is not None:
        future_rows = simulate_control(future_baseline, setpoints, deepcopy(states),
                                        date=today, seed_ctrl=seed_ctrl)
    else:
        # 스냅샷이 하나도 없으면 기존 경로(load_last_ctrl 시드 + fallback_clamp=True)와 동일 거동.
        initial_ctrl = load_last_ctrl()
        future_rows = simulate_control(future_baseline, setpoints, deepcopy(states),
                                        date=today, initial_ctrl=initial_ctrl, fallback_clamp=True)

    return past_rows + future_rows


# ── 상태 파일 (data/control_live_state.json) — 세션이 아닌 파일 기반 dedup ──────────

def _load_state() -> dict:
    from control import state_io
    return state_io.load_json(STATE_PATH)


def load_last_ctrl() -> "dict | None":
    """상태 파일에서 last_ctrl(직전 실행의 "제어 후" 값 시드)만 공개 API로 노출한다 —
    호출부(app/views/monitor.py 등)가 내부 상태 파일 포맷(_load_state())에 직접
    결합되지 않도록(이슈 #35 리뷰 P3). 읽기 실패해도 예외 전파 없이 None."""
    return _load_state().get("last_ctrl")


def _save_state(state: dict) -> None:
    """setpoints.save()와 동일한 원자적 쓰기 패턴 — 실패해도 예외 전파 없음."""
    from control import state_io
    state_io.save_json_atomic(STATE_PATH, state)


def _device_embed(transitions: dict, date_str: str) -> dict:
    from control.actuators import DEVICE_LABEL_KR
    lines = [f"{DEVICE_LABEL_KR.get(d, d)} {'ON' if on else 'OFF'}" for d, on in transitions.items()]
    return {"title": "🎛 오늘 운영 — 장치 전환", "color": 3447003,
            "fields": [{"name": "일시", "value": date_str, "inline": True},
                       {"name": "전환 내역", "value": "\n".join(lines) or "-"}]}


def _emergency_embed(hour_item: dict, date_str: str) -> dict:
    return {"title": "🚨 오늘 운영 — 긴급", "color": 15158332,
            "fields": [{"name": "일시", "value": f"{date_str} {hour_item['hour']:02d}시", "inline": True},
                       {"name": "사유", "value": hour_item["reason"]}]}


def _forecast_alert_embed(hour_items: "list[dict]", date_str: str) -> dict:
    """미래 시간대 사전 경보(이슈 #38 A안) — 여러 시간대를 1건 embed로 요약해 스팸을
    줄인다. hour_items는 hour 오름차순으로 넘겨야 fields 순서가 시간순이 된다."""
    fields = [{"name": f"{date_str} {item['hour']:02d}시", "value": item["reason"]}
              for item in hour_items]
    return {"title": "🔮 오늘 운영 — 사전 경보(예보 기반 예상)", "color": 15105642,
            "fields": fields}


def _kma_unavailable_embed(reason: str, date_str: str) -> dict:
    return {"title": "⚠️ 오늘 운영 — KMA 조회 실패", "color": 15105570,
            "fields": [{"name": "일시", "value": date_str, "inline": True},
                       {"name": "사유", "value": reason}]}


def run_notify(dry_run: bool = False, today: "_date | None" = None, now: "datetime | None" = None) -> int:
    """타이머 진입점(1시간마다 호출 가정) — 현재 시각 판정 → 직전 상태 파일과 비교해
    전환분만 디스코드로 발송(①장치 ON/OFF 전환 ②KMA 연속 실패 이상 ③긴급).

    장치 전환 판정은 timeline의 "지금"(hour==now.hour, 없으면 timeline 마지막) 항목
    기준이다 — render_live_tab()과 동일한 기준(P1-2: 타임라인 끝=미래 예보 기반 상태를
    "지금 전환"으로 오인해 발송하던 버그 수정).
    now 는 테스트 주입용(기본: 현재 시각).

    dry_run=True면 실제 발송 대신 stdout에 embed를 출력(로컬 수동 확인용).
    날짜가 바뀌면 상태가 자연히 리셋된다(전날 devices/emergency_hours를 베이스로 쓰지 않음).
    반환: 발송(혹은 dry-run 출력) 건수.
    """
    from control import setpoints as setpoints_mod
    from control.actuators import default_states
    from llm import notify

    today = today or datetime.now().date()
    date_str = today.isoformat()
    prev_state = _load_state()
    same_day = prev_state.get("date") == date_str

    def _emit(embed: dict) -> bool:
        if dry_run:
            print(f"[dry-run] {embed}")
            return True
        ok, _msg = notify.send_discord(embed)
        return ok

    outdoor = today_outdoor(today=today)
    sent = 0

    if outdoor is None:
        fail_count = (prev_state.get("fail_count", 0) + 1) if same_day else 1
        if fail_count >= _FAIL_ALERT_THRESHOLD:
            if _emit(_kma_unavailable_embed("KMA 외기 조회 연속 실패", date_str)):
                sent += 1
        if not same_day and prev_state.get("date"):
            archive_snapshots(prev_state)
        new_state = {"date": date_str, "fail_count": fail_count,
                     "devices": prev_state.get("devices", {}) if same_day else {},
                     "emergency_hours": prev_state.get("emergency_hours", []) if same_day else [],
                     "last_ctrl": prev_state.get("last_ctrl"),
                     "version": 2,
                     # snapshots도 보존 — KMA 실패로 판정을 못 해도 이미 기록된 오늘 스냅샷은
                     # 유실되면 안 된다(과거 픽스 전엔 new_state에서 빠져 유실되던 버그).
                     "snapshots": prev_state.get("snapshots", {}) if same_day else {}}
        if not dry_run:
            _save_state(new_state)
        return sent

    sp = setpoints_mod.load()
    states = default_states()
    baseline = indoor_baseline(outdoor, date=today)
    # 자정 연속성(이슈 #35) — 직전 실행이 남긴 "제어 후" 값을 시드로 이어받아 0시 불연속을
    # 완화한다(prev_state["last_ctrl"], 없거나 무효하면 simulate_control이 밴드 클램프 폴백).
    initial_ctrl = prev_state.get("last_ctrl")
    timeline = simulate_control(baseline, sp, states, date=today,
                                 initial_ctrl=initial_ctrl, fallback_clamp=True)
    emg = emergency_hours(timeline, sp)

    # 장치 전환 판정은 타임라인 마지막(=미래 예보 기반 마지막 시간) states가 아니라 "지금"에
    # 해당하는 시간대 항목 기준이어야 한다(P1-2) — render_live_tab()과 동일한 기준(hour==현재
    # 시각, 없으면 timeline 마지막)을 사용해 UI와 알림이 같은 "지금" 상태를 보게 맞춘다.
    now_hour = (now or datetime.now()).hour
    cur_item = next((t for t in timeline if t["hour"] == now_hour), timeline[-1] if timeline else None)
    cur_devices_on = set(cur_item["devices_on"]) if cur_item else set()
    cur_devices = {d: (d in cur_devices_on) for d in default_states()}

    prev_devices = prev_state.get("devices", {}) if same_day else {}
    transitions = {d: on for d, on in cur_devices.items() if prev_devices.get(d, False) != on}
    if transitions:
        if _emit(_device_embed(transitions, date_str)):
            sent += 1

    prev_emg_keys = set(prev_state.get("emergency_hours", [])) if same_day else set()
    # cur_emg_keys는 emg 전체(현재/미래/과거 불문)로 만든다 — 과거 시간대는 발송하지
    # 않아도 상태 키에는 남겨야 재발송(뒷북)을 막을 수 있다(이슈 #38 A안).
    cur_emg_keys = {_emergency_key(item) for item in emg}
    new_emg = [item for item in emg if _emergency_key(item) not in prev_emg_keys]

    # 시점별 분기(이슈 #38 A안): 현재 시각(hour==now_hour)만 즉시 "🚨 긴급", 미래
    # (hour>now_hour)는 "🔮 사전 경보"로 묶어 1건만 발송, 과거(hour<now_hour)는 뒷북이라
    # 발송하지 않는다(단, 위에서 cur_emg_keys에는 이미 포함돼 재판정을 막는다).
    current_emg = [item for item in new_emg if item["hour"] == now_hour]
    future_emg = sorted((item for item in new_emg if item["hour"] > now_hour),
                         key=lambda item: item["hour"])

    for item in current_emg:
        if _emit(_emergency_embed(item, date_str)):
            sent += 1
    if future_emg:
        if _emit(_forecast_alert_embed(future_emg, date_str)):
            sent += 1

    last_ctrl = ({"date": date_str, "hour": cur_item["hour"], "temp": cur_item["ctrl_temp"],
                  "hum": cur_item["ctrl_hum"]} if cur_item else prev_state.get("last_ctrl"))

    # 시간별 스냅샷 누적(이슈 #40) — 같은 날이면 직전 상태의 snapshots를 이어받고, 날짜가
    # 바뀌면 그 전날 것을 history로 아카이브한 뒤 새로 시작한다.
    if same_day:
        snapshots = dict(prev_state.get("snapshots", {}))
    else:
        if prev_state.get("date"):
            archive_snapshots(prev_state)
        snapshots = {}
    new_state = {"date": date_str, "fail_count": 0, "devices": cur_devices,
                 "emergency_hours": sorted(cur_emg_keys), "last_ctrl": last_ctrl,
                 "version": 2, "snapshots": snapshots}
    if cur_item:
        record_snapshot(new_state, cur_item, source="sim")
    if not dry_run:
        _save_state(new_state)
    return sent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="오늘 운영 모드 — 전환분만 디스코드 알림(타이머 진입점)")
    ap.add_argument("--notify", action="store_true", help="1회 판정 후 전환분 알림 발송")
    ap.add_argument("--dry-run", action="store_true", help="발송 대신 stdout에 embed 출력")
    args = ap.parse_args()
    if args.notify:
        n = run_notify(dry_run=args.dry_run)
        print(f"[control.live] 처리 완료 — {n}건 {'dry-run 출력' if args.dry_run else '발송'}")
    else:
        ap.print_help()
