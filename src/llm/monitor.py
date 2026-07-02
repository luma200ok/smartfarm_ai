"""
센서 자동 감시 알림 — 규칙 임계값을 넘으면 디스코드 웹훅으로 자동 발송.

가상 센서(sim.virtual_sensor) 스트림을 하루씩 감시하다 위험 임계를 넘으면 알림.
중복 방지: 같은 위험이 지속되면 재발송하지 않고 '상태 진입' 시에만 1회 발송.
ML/DL과 같은 피처(infer.ENV_FEATURES)·진단 라벨(leaf_mold 등)로 말한다(라벨 동기화).
self-contained — 앱·스프링 서버와 무관, 웹훅만 공유.

실행:  python src/llm/monitor.py --year 2024 --interval 1 [--loop] [--llm]
전제:  .env의 DISCORD_WEBHOOK_URL (없으면 감지 로그만, 발송은 '미설정')
"""
import argparse
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dl import infer  # noqa: E402
from llm import expect as expect_module  # noqa: E402
from llm import history, notify  # noqa: E402
from sim.virtual_sensor import VirtualSensor  # noqa: E402

_log = logging.getLogger(__name__)

# 규칙 임계값 (조정 가능) — 피처명은 infer.ENV_FEATURES와 동일
HUM_CRIT, HUM_WARN = 90.0, 85.0        # 습도내부_평균(%): 곰팡이 위험
TEMP_HOT, TEMP_COLD = 35.0, 5.0        # 온도내부_평균(℃): 고온/냉해
HUM_DISEASES = ["leaf_mold", "late_blight"]  # 이슈 #2 흡수 — 고습은 두 병해군 모두 위험
FORECAST_MARGIN = 2.0                  # feedforward: 기대 최저 < TEMP_COLD+margin 이면 사전 경보
EQUIP_WARN_Z, EQUIP_CRIT_Z = -2.0, -3.0  # equip_anom 판정 z-score(잔차/σ)
_LEVEL_COLOR = {"경고": 15158332, "주의": 15105570}   # 빨강/주황
_GRAY = 9807270


def _cause(z: float, cold: bool) -> str:
    """잔차 z-score(=(실측-기대)/σ) → 원인 추정 문구.

    cold=True(저온 경보 방향): |z|<2 → 기대값 자체가 낮음(외기 요인) / z<=-2 → 기대보다도
    더 낮음(설비 고장 의심). cold=False(고온 경보 방향)는 대칭(냉방)."""
    if cold:
        return "설비(난방) 고장 의심" if z <= -2.0 else "외기 요인(한파로 설비 능력 한계)"
    return "설비(냉방) 고장 의심" if z >= 2.0 else "외기 요인(폭염으로 설비 능력 한계)"


def assess(reading: dict, expect: dict | None = None) -> list[dict]:
    """센서 관측값 → 위험 목록 [{key, level, disease, reason, ...}]. 없으면 [].

    expect(옵션, expect.expected()의 반환값)를 주면 temp 경보에 cause(원인 추정)를 붙이고,
    임계 미달 상태에서도 잔차가 -2σ/-3σ 이면 별도 equip_anom 경보를 추가한다.
    expect=None(기본)이면 기존 동작 그대로(하위호환)."""
    hum = reading.get("습도내부_평균", 0.0)
    temp = reading.get("온도내부_평균", 20.0)
    out = []
    if hum >= HUM_CRIT:
        out.append({"key": "humidity", "level": "경고", "disease": "leaf_mold",
                    "diseases": list(HUM_DISEASES),
                    "reason": f"고습({hum:.0f}%≥{HUM_CRIT:.0f}) — 잎곰팡이병·잎마름역병 급속 확산 위험"})
    elif hum >= HUM_WARN:
        out.append({"key": "humidity", "level": "주의", "disease": "leaf_mold",
                    "diseases": list(HUM_DISEASES),
                    "reason": f"습도 높음({hum:.0f}%≥{HUM_WARN:.0f}) — 곰팡이·역병 주의, 환기 권장"})

    z = None
    if expect is not None and "평균" in expect:
        sigma = (expect.get("resid_sigma") or {}).get("평균")
        z = expect_module.z_score(temp, expect["평균"], sigma) if sigma else None

    if temp >= TEMP_HOT:
        alert = {"key": "temp_hot", "level": "경고", "disease": "",
                 "reason": f"고온({temp:.0f}℃≥{TEMP_HOT:.0f}) — 착과·생육 장해 위험"}
        if z is not None:
            alert["cause"] = _cause(z, cold=False)
        out.append(alert)
    elif temp <= TEMP_COLD:
        alert = {"key": "temp_cold", "level": "경고", "disease": "",
                 "reason": f"저온({temp:.0f}℃≤{TEMP_COLD:.0f}) — 냉해 위험"}
        if z is not None:
            alert["cause"] = _cause(z, cold=True)
        out.append(alert)
    elif z is not None:
        # 임계(TEMP_COLD) 미달 전이라도 잔차가 크게 음수면 설비 이상 의심(별도 경보)
        level = "경고" if z <= EQUIP_CRIT_Z else ("주의" if z <= EQUIP_WARN_Z else None)
        if level:
            out.append({"key": "equip_anom", "level": level, "disease": "",
                        "cause": _cause(z, cold=True),
                        "reason": f"내부 기대값 대비 급락(z={z:.1f}σ) — 임계 도달 전이지만 설비 점검 권장"})
    return out


def _akey(a: dict) -> str:
    """dedup 식별자 — 위험종류+레벨. 레벨이 바뀌면(주의→경고 악화) 다른 키 → 재발송."""
    return f"{a['key']}:{a['level']}"


def evaluate(reading: dict, active: set, expect: dict | None = None) -> tuple[list, set]:
    """순수 함수 — 새로 진입한 위험만 발송 대상. (to_send, 현재 활성 키셋). 중복 방지 핵심.

    식별자에 레벨을 포함해 '지속'은 억제하되 '심각도 변화'는 새 알림으로 취급.
    """
    alerts = assess(reading, expect)
    keys = {_akey(a) for a in alerts}
    to_send = [a for a in alerts if _akey(a) not in active]
    return to_send, keys


def feedforward_alerts(forecast_daily: list[dict], expect_model=None, active: set | None = None) -> list[dict]:
    """예보(또는 리플레이 선견) 기반 사전 경보 — 순수 함수.

    forecast_daily: [{"date":..., "tmn":..., "tmx":...}, ...] (weather.get_forecast_3d()["daily"]
    형식과 동일. tmx 없으면 tmn을 외기로 사용).
    expect_model: expect.load_model()이 반환하는 payload(dict). None이면 내부에서 lazy-load.
    active: 이미 활성인 dedup 키 집합(monitor.evaluate 와 동일한 set 공유 가능).
    반환: temp_cold 경보와 동일한 형태의 alert 목록(key는 forecast_cold:{date}로 유일).
    """
    active = active or set()
    model = expect_model if expect_model is not None else expect_module.load_model()
    if model is None:
        return []
    clim = model.get("doy_solar_climatology") or {}
    out = []
    for d in forecast_daily:
        date, tmn, tmx = d.get("date"), d.get("tmn"), d.get("tmx")
        if date is None or tmn is None:
            continue
        outer = (tmn + tmx) / 2.0 if tmx is not None else tmn
        n = expect_module.doy(date)
        solar = clim.get(n, clim.get(str(n)))
        if solar is None:
            continue
        pred = expect_module.predict(model, {"온도외부_평균": outer, "일사량_평균": solar}, date)
        if pred is None or "최저" not in pred:
            continue
        if pred["최저"] < TEMP_COLD + FORECAST_MARGIN:
            key = f"forecast_cold:{date}"
            if f"{key}:주의" in active:
                continue
            out.append({"key": key, "level": "주의", "disease": "",
                        "reason": f"{date} 야간 최저 기대 {pred['최저']:.1f}℃"
                                  f"(<{TEMP_COLD + FORECAST_MARGIN:.0f}℃) — 사전 대비(보온·난방 점검) 권장"})
    return out


def _embed(alert: dict, reading: dict, date: str, use_llm: bool = False, window=None) -> dict:
    fields = [
        {"name": "일시", "value": str(date), "inline": True},
        {"name": "레벨", "value": alert["level"], "inline": True},
        {"name": "측정",
         "value": f"내부온도 {reading['온도내부_평균']:.1f}℃ · 습도 {reading['습도내부_평균']:.0f}%"},
        {"name": "사유", "value": notify._t(alert["reason"])},
    ]
    if alert.get("diseases"):
        names = "·".join(infer.LABEL_KR.get(d, d) for d in alert["diseases"])
        fields.append({"name": "관련 병해", "value": names, "inline": True})
    elif alert.get("disease"):
        fields.append({"name": "관련 병해",
                       "value": infer.LABEL_KR.get(alert["disease"], alert["disease"]), "inline": True})
    if alert.get("cause"):
        fields.append({"name": "추정 원인", "value": alert["cause"], "inline": True})
    if use_llm and window is not None:
        try:                                       # 위험 첫 발생 시 LLM 보강(RAG 근거)
            from llm.pipeline import early_warning
            w = early_warning(window)
            fields.append({"name": "AI 경보", "value": notify._t(f"{w.이유} → {w.권장조치}")})
        except Exception as e:
            _log.debug("LLM 보강 실패(무시): %s", e)   # 발송은 그대로 진행
    return {"title": f"🚨 스마트팜 자동 경보 — {alert['level']}",
            "color": _LEVEL_COLOR.get(alert["level"], _GRAY), "fields": fields}


def run(year=None, interval=1.0, loop=False, use_llm=False, max_days=None, feedforward=False) -> int:
    vs = VirtualSensor(year)
    total = max(1, len(vs.series) - (infer.WINDOW - 1))   # 커서 시작(WINDOW-1) 보정 = 고유 일수
    print(f"[monitor] {vs.farm} · {total}일 재생 · interval={interval}s · loop={loop}")
    model = expect_module.load_model()      # pkl 없으면 None — 이하 전부 우아한 저하
    active: set = set()
    steps = 0
    while True:
        reading, date = vs.reading(), vs.date()
        window = vs.window() if use_llm else None
        exp = expect_module.predict(model, reading, date) if model is not None else None
        to_send, active = evaluate(reading, active, exp)
        if feedforward and model is not None:
            try:
                from llm import weather
                fcst = weather.get_forecast_3d()
                daily = (fcst or {}).get("daily") or []
                ff_alerts = feedforward_alerts(daily, expect_model=model, active=active)
            except Exception as e:                       # best-effort — 실패해도 본감시는 계속
                _log.debug("feedforward 조회 실패(무시): %s", e)
                ff_alerts = []
            for a in ff_alerts:
                active.add(_akey(a))
            to_send = to_send + ff_alerts
        print(f"  {date} | 내부 {reading['온도내부_평균']:.1f}℃ 습도 {reading['습도내부_평균']:.0f}% "
              f"| {','.join(sorted(active)) or '정상'}")
        for a in to_send:
            ok, msg = notify.send_discord(_embed(a, reading, date, use_llm, window))
            if not ok:
                active.discard(_akey(a))              # 전송 실패 → 다음 틱 재시도(유실 방지)
            history.save_alert("monitor", a["level"], a.get("disease", ""), a["reason"],
                               {**a, "reading": reading, "date": str(date), "sent": ok})
            print(f"    🚨 [{a['level']}] {a['reason']} → 디스코드 {'전송됨' if ok else msg}")
        steps += 1
        if (max_days and steps >= max_days) or (not loop and steps >= total):
            break
        vs.tick()
        if interval:
            time.sleep(interval)
    print(f"[monitor] 종료 — {steps}일 감시")
    return steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="센서 자동 감시 → 디스코드 알림")
    ap.add_argument("--year", type=int, default=None, help="재생 작기(연도 라벨), 미지정 시 최신")
    ap.add_argument("--interval", type=float, default=1.0, help="틱 간격(초)")
    ap.add_argument("--loop", action="store_true", help="끝나면 처음부터 반복(상주)")
    ap.add_argument("--llm", action="store_true", help="위험 첫 발생 시 LLM 문구 보강")
    ap.add_argument("--feedforward", action="store_true", help="KMA 3일 예보 기반 사전 경보 병행")
    ap.add_argument("--max", type=int, default=None, help="최대 감시 일수(데모)")
    a = ap.parse_args()
    run(year=a.year, interval=a.interval, loop=a.loop, use_llm=a.llm, max_days=a.max, feedforward=a.feedforward)
