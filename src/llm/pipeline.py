"""
Phase 3-3 — 재배 도우미 loop 중 사진 없는 흐름: 일일코치(A)·조기경보(B).

사진→처방(C)은 prescribe.py. 여기(A·B)는 LSTM 환경예측(get_forecast)을 근거로:
- daily_coach(): 평상시 "오늘 할 일" 코칭
- early_warning(): 환경(습도위험)×병해위험 → 선제 점검 경보 (고습이면 RAG 잎곰팡이병 근거)

실행:  python src/llm/pipeline.py    (Ollama + env_lstm.pt 필요)
"""
import json
import logging
import sys
from pathlib import Path

import ollama
from pydantic import BaseModel, Field, ValidationError

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import history  # noqa: E402
from llm.prescribe import MODEL  # noqa: E402  (동일 Ollama 모델·.env 로드 재사용)
from llm.rag import retrieve  # noqa: E402
from llm.tools import TOOL_REGISTRY, get_forecast  # noqa: E402

_log = logging.getLogger(__name__)

WEATHER_QA_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "기상청(KMA) 외부 날씨를 조회한다. kind='current'면 현재 실황(기온·습도·강수), "
                           "kind='forecast'(기본)면 3일 예보(날짜별 최저·최고기온, 시간별 기온·습도·강수확률·하늘상태)를 반환.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["current", "forecast"],
                             "description": "current=현재 실황, forecast=3일 예보(기본)"},
                },
            },
        },
    },
]

WEATHER_QA_SYSTEM = (
    "너는 토마토 온실 재배를 돕는 한국어 재배 도우미다. 사용자가 외부 날씨(기온·습도·강수·예보)를 물으면 "
    "get_weather tool을 호출해 실제 수치를 확인하고, 그 수치를 토마토 온실 관점에서 해석해 답하라 "
    "(예: '내일 최저 3°C 급락, 보온 조치 권장'). 근거 없는 수치는 지어내지 말고, tool 결과가 "
    "unavailable이면 조회가 안 됐다고 정직히 안내하라. 자연스러운 한국어 문장으로 답하고 JSON은 쓰지 말라."
)

WEATHER_QA_MAX_ROUNDS = 2
WEATHER_QA_UNAVAILABLE_MSG = "날씨 정보를 지금 가져올 수 없어요. 잠시 후 다시 시도해 주세요."

COACH_SYSTEM = (
    "너는 토마토 재배 초보자를 돕는 한국어 재배 코치다. 환경 예측 수치를 바탕으로 "
    "오늘 챙길 관리 작업을 쉽고 구체적으로 알려준다. 지정된 JSON 스키마로만 답하라."
)
WARN_SYSTEM = (
    "너는 토마토 병해 조기경보 도우미다. 환경 예측(특히 습도위험)과 제공된 재배가이드 근거로 "
    "병해 발생 위험을 판단해 선제 점검을 안내한다. 위험이 낮으면 경보수준을 '정상'으로 하라. "
    "근거에 없는 내용은 지어내지 말고, 지정된 JSON 스키마로만 답하라."
)


class Coach(BaseModel):
    요약: str = Field(description="오늘 환경 상태 한 줄 요약")
    오늘_할일: list[str] = Field(description="오늘 해야 할 관리 작업 2~4개")
    근거: str = Field(description="환경 예측 수치에 근거한 이유")


class Warning(BaseModel):
    경보수준: str = Field(description="정상 / 주의 / 경고 중 하나")
    위험병해: str = Field(description="주의할 병해 이름, 없으면 '없음'")
    이유: str = Field(description="이 경보수준으로 판단한 이유")
    권장조치: str = Field(description="지금 권장하는 선제 조치")


def _forecast_text(fc: dict | None) -> str:
    if not fc or fc.get("unavailable"):
        return "환경 예측 데이터 없음(센서/모델 미연동)"
    return (f"다음날 내부온도 {fc['next_temp']}℃({fc['trend']}), "
            f"최근 습도 평균 {fc['humidity_mean']}% → 습도위험 '{fc['humidity_risk']}'")


def daily_coach(window=None) -> Coach:
    """A — 평상시 일일 코치. 환경 예측 기반 '오늘 할 일'. window 명시 시 그 시점 기준(가상 센서)."""
    fc = get_forecast(window)
    msg = [{"role": "system", "content": COACH_SYSTEM},
           {"role": "user", "content": f"오늘의 토마토 재배 코칭을 해줘.\n환경 예측: {_forecast_text(fc)}"}]
    resp = ollama.chat(model=MODEL, messages=msg, format=Coach.model_json_schema())
    try:
        return Coach.model_validate_json(resp["message"]["content"])
    except ValidationError as e:
        _log.warning("Coach 스키마 검증 실패: %s", e)
        return Coach(요약="코칭 생성 실패", 오늘_할일=[], 근거="-")


def early_warning(window=None) -> Warning:
    """B — 조기 경보. 습도위험 높으면 잎곰팡이병 위험 경보(+RAG 근거). window 명시 시 그 시점 기준."""
    fc = get_forecast(window)
    if not fc or fc.get("unavailable"):
        return Warning(경보수준="정상", 위험병해="없음",
                       이유="환경 예측 데이터가 없어 판단 불가", 권장조치="센서/모델 연동 후 재확인")
    risk = fc.get("humidity_risk")
    chunks = retrieve("고습 환경 잎곰팡이병 예방", disease="leaf_mold", k=2) if risk in ("높음", "보통") else []
    guide = "\n".join(c["text"] for c in chunks) or "(해당 위험 근거 없음)"
    msg = [{"role": "system", "content": WARN_SYSTEM},
           {"role": "user", "content": f"환경 예측: {_forecast_text(fc)}\n\n재배가이드 근거:\n{guide}\n\n"
                                        "이 조건에서 토마토 병해 조기경보를 판단해줘."}]
    resp = ollama.chat(model=MODEL, messages=msg, format=Warning.model_json_schema())
    try:
        w = Warning.model_validate_json(resp["message"]["content"])
        history.save_alert("early_warning", w.경보수준, w.위험병해, w.이유, w.model_dump())
        return w
    except ValidationError as e:
        _log.warning("Warning 스키마 검증 실패: %s", e)
        return Warning(경보수준="정상", 위험병해="없음", 이유="판단 실패", 권장조치="잠시 후 재시도")


def weather_qa(user_msg: str) -> str:
    """날씨 단독 Q&A — get_weather tool 호출 루프(최대 2라운드) 후 일반 텍스트 응답.

    처방 흐름(prescribe.py)과 완전히 별개 경로. Ollama 미가동/tool unavailable이어도
    예외를 전파하지 않고 안내 문구를 반환한다.
    """
    messages = [{"role": "system", "content": WEATHER_QA_SYSTEM},
                {"role": "user", "content": user_msg}]
    try:
        for _ in range(WEATHER_QA_MAX_ROUNDS):
            resp = ollama.chat(model=MODEL, messages=messages, tools=WEATHER_QA_TOOL_SCHEMA)
            msg = resp["message"]
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                content = (msg.get("content") or "").strip()
                return content or WEATHER_QA_UNAVAILABLE_MSG
            for tc in calls:
                name = tc["function"]["name"]
                args = dict(tc["function"]["arguments"] or {})
                fn = TOOL_REGISTRY.get(name)
                if fn is None:
                    result = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = {"unavailable": True, "reason": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, ensure_ascii=False)})
        # 마지막 라운드까지 tool 호출만 반복된 경우 — 지금까지의 메시지로 최종 답 한 번 더 요청
        final = ollama.chat(model=MODEL, messages=messages)
        content = (final["message"].get("content") or "").strip()
        return content or WEATHER_QA_UNAVAILABLE_MSG
    except Exception as e:
        _log.warning("weather_qa 실패: %s", e)
        return WEATHER_QA_UNAVAILABLE_MSG


if __name__ == "__main__":
    import json
    print(f"[모델] {MODEL}\n[환경예측] {_forecast_text(get_forecast())}\n")
    print("=== 🌅 일일 코치 (A) ===")
    print(json.dumps(daily_coach().model_dump(), ensure_ascii=False, indent=2))
    print("\n=== ⚠️ 조기 경보 (B) ===")
    print(json.dumps(early_warning().model_dump(), ensure_ascii=False, indent=2))
