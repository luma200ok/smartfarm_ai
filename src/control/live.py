"""오늘 운영 모드(이슈 #23) — KMA 외기 + 기대값 모델로 오늘 시간대별
"제어 전 내부 기준선"과 "장치 효과가 반영된 제어 후 내부값"을 계산한다.

규칙 기반(controller.decide()·effects.EFFECTS 재사용), LLM 호출 없음.
run_notify()는 서버 타이머(1시간 간격, systemd timer)에서 호출되는 진입점 —
장치 ON/OFF 전환·긴급·이상만 디스코드로 발송(상태 파일 dedup, 세션이 아니라
파일 기반이라 앱·타이머 프로세스가 상태를 공유한다).
"""
import argparse
import json
import logging
import os
from datetime import date as _date
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
ROOT = _SRC.parent

_log = logging.getLogger(__name__)

STATE_PATH = ROOT / "data" / "control_live_state.json"

# 장치 효과(effects.EFFECTS)는 "1일치 풀가동" 델타로 정의돼 있다 — 시간 단위 시뮬레이션에서는
# 1시간 ON일 때 그 델타의 1/24만 반영되도록 나눈다(선형 근사, 정밀 열역학 모델 아님).
HOURLY_EFFECT_DIVISOR = 24.0

# 실내 습도 기준선 근사 — 외기습도 + 온실 보습 보정(상수, 실측 아닌 근사치).
INDOOR_HUMIDITY_OFFSET = 10.0

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
    내부 습도 기준선은 외기 습도 + INDOOR_HUMIDITY_OFFSET(온실 보습 보정, 상수)로 단순
    근사한다 — 둘 다 정밀 물리 모델이 아니라 대시보드용 근사치임에 유의.
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
        if humidity is not None:
            base_hum = min(humidity + INDOOR_HUMIDITY_OFFSET, 100.0)

        baseline.append({
            "hour": hour, "out_temp": temp, "out_hum": humidity,
            "base_temp": base_temp, "base_hum": base_hum,
        })
    return baseline


def simulate_control(baseline: "list[dict]", setpoints, states, date=None) -> "list[dict]":
    """시간 순 재생 — controller.decide() 재사용, ON 장치 효과(시간 환산)를 다음 시간
    내부값에 반영해 "제어 후" 내부온·습도를 산출한다.

    반환: [{hour, out_temp, base_temp, ctrl_temp, base_hum, ctrl_hum, devices_on, events}]
    (events = 그 시간대에 발생한 ControlLog 목록).
    """
    from control import controller
    from control.effects import EFFECTS

    date = date or datetime.now().date()
    timeline = []
    ctrl_temp = None
    ctrl_hum = None

    for idx, item in enumerate(baseline):
        if ctrl_temp is None:
            ctrl_temp = item["base_temp"]
        if ctrl_hum is None:
            ctrl_hum = item["base_hum"]

        reading = {"온도내부_평균": ctrl_temp, "습도내부_평균": ctrl_hum}
        logs = controller.decide(reading, setpoints, states, date=f"{date} {item['hour']:02d}:00")
        devices_on = [d for d in states if states[d].on]

        timeline.append({
            "hour": item["hour"], "out_temp": item["out_temp"], "base_temp": item["base_temp"],
            "ctrl_temp": ctrl_temp, "base_hum": item["base_hum"], "ctrl_hum": ctrl_hum,
            "devices_on": devices_on, "events": logs,
        })

        delta_temp = sum(EFFECTS.get(d, {}).get("온도내부_평균", 0.0) for d in devices_on) / HOURLY_EFFECT_DIVISOR
        delta_hum = sum(EFFECTS.get(d, {}).get("습도내부_평균", 0.0) for d in devices_on) / HOURLY_EFFECT_DIVISOR

        next_base_temp = baseline[idx + 1]["base_temp"] if idx + 1 < len(baseline) else None
        next_base_hum = baseline[idx + 1]["base_hum"] if idx + 1 < len(baseline) else None

        base_for_next_temp = next_base_temp if next_base_temp is not None else ctrl_temp
        ctrl_temp = (base_for_next_temp + delta_temp) if base_for_next_temp is not None else None

        base_for_next_hum = next_base_hum if next_base_hum is not None else ctrl_hum
        ctrl_hum = (base_for_next_hum + delta_hum) if base_for_next_hum is not None else None

    return timeline


def emergency_hours(timeline: "list[dict]", setpoints) -> "list[dict]":
    """풀가동(관련 장치 전부 ON)에도 제어 후 값이 밴드 밖인 시간대 → [{hour, reason}, ...]."""
    out = []
    for item in timeline:
        temp = item["ctrl_temp"]
        hum = item["ctrl_hum"]
        devices_on = set(item["devices_on"])
        if temp is not None:
            if temp > setpoints.temp_high and {"cooling_fan", "vent"} <= devices_on:
                out.append({"hour": item["hour"],
                            "reason": f"제어 한계 초과 — 냉방·환기 풀가동에도 고온 지속({temp:.1f}℃)"})
            elif temp < setpoints.temp_low and "heater" in devices_on:
                out.append({"hour": item["hour"],
                            "reason": f"제어 한계 초과 — 난방 풀가동에도 저온 지속({temp:.1f}℃)"})
        if hum is not None:
            if hum > setpoints.hum_high and "vent" in devices_on:
                out.append({"hour": item["hour"],
                            "reason": f"제어 한계 초과 — 환기 풀가동에도 고습 지속({hum:.1f}%)"})
            elif hum < setpoints.hum_low and "humidifier" in devices_on:
                out.append({"hour": item["hour"],
                            "reason": f"제어 한계 초과 — 가습 풀가동에도 저습 지속({hum:.1f}%)"})
    return out


# ── 상태 파일 (data/control_live_state.json) — 세션이 아닌 파일 기반 dedup ──────────

def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    """setpoints.save()와 동일한 원자적 쓰기 패턴 — 실패해도 예외 전파 없음."""
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except OSError:
        return


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


def _kma_unavailable_embed(reason: str, date_str: str) -> dict:
    return {"title": "⚠️ 오늘 운영 — KMA 조회 실패", "color": 15105570,
            "fields": [{"name": "일시", "value": date_str, "inline": True},
                       {"name": "사유", "value": reason}]}


def run_notify(dry_run: bool = False, today: "_date | None" = None) -> int:
    """타이머 진입점(1시간마다 호출 가정) — 현재 시각 판정 → 직전 상태 파일과 비교해
    전환분만 디스코드로 발송(①장치 ON/OFF 전환 ②KMA 연속 실패 이상 ③긴급).

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
        new_state = {"date": date_str, "fail_count": fail_count,
                     "devices": prev_state.get("devices", {}) if same_day else {},
                     "emergency_hours": prev_state.get("emergency_hours", []) if same_day else []}
        if not dry_run:
            _save_state(new_state)
        return sent

    sp = setpoints_mod.load()
    states = default_states()
    baseline = indoor_baseline(outdoor, date=today)
    timeline = simulate_control(baseline, sp, states, date=today)
    emg = emergency_hours(timeline, sp)

    prev_devices = prev_state.get("devices", {}) if same_day else {}
    cur_devices = {d: s.on for d, s in states.items()}
    transitions = {d: on for d, on in cur_devices.items() if prev_devices.get(d, False) != on}
    if transitions:
        if _emit(_device_embed(transitions, date_str)):
            sent += 1

    prev_emg_keys = set(prev_state.get("emergency_hours", [])) if same_day else set()
    cur_emg_keys = {str(item["hour"]) for item in emg}
    new_emg = [item for item in emg if str(item["hour"]) not in prev_emg_keys]
    for item in new_emg:
        if _emit(_emergency_embed(item, date_str)):
            sent += 1

    new_state = {"date": date_str, "fail_count": 0, "devices": cur_devices,
                 "emergency_hours": sorted(cur_emg_keys)}
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
