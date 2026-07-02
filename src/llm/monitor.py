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
from llm import history, notify  # noqa: E402
from sim.virtual_sensor import VirtualSensor  # noqa: E402

_log = logging.getLogger(__name__)

# 규칙 임계값 (조정 가능) — 피처명은 infer.ENV_FEATURES와 동일
HUM_CRIT, HUM_WARN = 90.0, 85.0        # 습도내부_평균(%): 곰팡이 위험
TEMP_HOT, TEMP_COLD = 35.0, 5.0        # 온도내부_평균(℃): 고온/냉해
_LEVEL_COLOR = {"경고": 15158332, "주의": 15105570}   # 빨강/주황
_GRAY = 9807270


def assess(reading: dict) -> list[dict]:
    """센서 관측값 → 위험 목록 [{key, level, disease, reason}]. 없으면 []."""
    hum = reading.get("습도내부_평균", 0.0)
    temp = reading.get("온도내부_평균", 20.0)
    out = []
    if hum >= HUM_CRIT:
        out.append({"key": "humidity", "level": "경고", "disease": "leaf_mold",
                    "reason": f"고습({hum:.0f}%≥{HUM_CRIT:.0f}) — 잎곰팡이병·잎마름역병 급속 확산 위험"})
    elif hum >= HUM_WARN:
        out.append({"key": "humidity", "level": "주의", "disease": "leaf_mold",
                    "reason": f"습도 높음({hum:.0f}%≥{HUM_WARN:.0f}) — 곰팡이·역병 주의, 환기 권장"})
    if temp >= TEMP_HOT:
        out.append({"key": "temp_hot", "level": "경고", "disease": "",
                    "reason": f"고온({temp:.0f}℃≥{TEMP_HOT:.0f}) — 착과·생육 장해 위험"})
    elif temp <= TEMP_COLD:
        out.append({"key": "temp_cold", "level": "경고", "disease": "",
                    "reason": f"저온({temp:.0f}℃≤{TEMP_COLD:.0f}) — 냉해 위험"})
    return out


def _akey(a: dict) -> str:
    """dedup 식별자 — 위험종류+레벨. 레벨이 바뀌면(주의→경고 악화) 다른 키 → 재발송."""
    return f"{a['key']}:{a['level']}"


def evaluate(reading: dict, active: set) -> tuple[list, set]:
    """순수 함수 — 새로 진입한 위험만 발송 대상. (to_send, 현재 활성 키셋). 중복 방지 핵심.

    식별자에 레벨을 포함해 '지속'은 억제하되 '심각도 변화'는 새 알림으로 취급.
    """
    alerts = assess(reading)
    keys = {_akey(a) for a in alerts}
    to_send = [a for a in alerts if _akey(a) not in active]
    return to_send, keys


def _embed(alert: dict, reading: dict, date: str, use_llm: bool = False, window=None) -> dict:
    fields = [
        {"name": "일시", "value": str(date), "inline": True},
        {"name": "레벨", "value": alert["level"], "inline": True},
        {"name": "측정",
         "value": f"내부온도 {reading['온도내부_평균']:.1f}℃ · 습도 {reading['습도내부_평균']:.0f}%"},
        {"name": "사유", "value": notify._t(alert["reason"])},
    ]
    if alert.get("disease"):
        fields.append({"name": "관련 병해",
                       "value": infer.LABEL_KR.get(alert["disease"], alert["disease"]), "inline": True})
    if use_llm and window is not None:
        try:                                       # 위험 첫 발생 시 LLM 보강(RAG 근거)
            from llm.pipeline import early_warning
            w = early_warning(window)
            fields.append({"name": "AI 경보", "value": notify._t(f"{w.이유} → {w.권장조치}")})
        except Exception as e:
            _log.debug("LLM 보강 실패(무시): %s", e)   # 발송은 그대로 진행
    return {"title": f"🚨 스마트팜 자동 경보 — {alert['level']}",
            "color": _LEVEL_COLOR.get(alert["level"], _GRAY), "fields": fields}


def run(year=None, interval=1.0, loop=False, use_llm=False, max_days=None) -> int:
    vs = VirtualSensor(year)
    total = max(1, len(vs.series) - (infer.WINDOW - 1))   # 커서 시작(WINDOW-1) 보정 = 고유 일수
    print(f"[monitor] {vs.farm} · {total}일 재생 · interval={interval}s · loop={loop}")
    active: set = set()
    steps = 0
    while True:
        reading, date = vs.reading(), vs.date()
        window = vs.window() if use_llm else None
        to_send, active = evaluate(reading, active)
        print(f"  {date} | 내부 {reading['온도내부_평균']:.1f}℃ 습도 {reading['습도내부_평균']:.0f}% "
              f"| {','.join(sorted(active)) or '정상'}")
        for a in to_send:
            ok, msg = notify.send_discord(_embed(a, reading, date, use_llm, window))
            if not ok:
                active.discard(_akey(a))              # 전송 실패 → 다음 틱 재시도(유실 방지)
            history.save_alert("monitor", a["level"], a["disease"], a["reason"],
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
    ap.add_argument("--max", type=int, default=None, help="최대 감시 일수(데모)")
    a = ap.parse_args()
    run(year=a.year, interval=a.interval, loop=a.loop, use_llm=a.llm, max_days=a.max)
