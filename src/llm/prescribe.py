"""
Phase 3-1 — Ollama(qwen2.5:14b) 자연어 처방 오케스트레이션.

LLM 주도 function calling: LLM이 get_diagnosis/get_detection tool을 스스로 호출 →
우리가 실제 DL 추론(src/llm/tools.py) 실행 → 결과를 받아 LLM이 처방 생성.

환각 방어 3종:
  ① 신뢰도 톤 분기   — 진단 prob 밴드(>0.8 단정 / 0.6~0.8 중립 / <0.6 정밀확인)로 지시문 주입
  ② 게이트 차단 안내 — ood_blocked면 병명 단정 금지, 재촬영 안내
  ③ 클래스 한정성    — 아는 범위(잎 병해 3종)를 system 프롬프트에 고정, 밖이면 '진단 보류'
최종 출력은 Prescription JSON 스키마로 강제(format=).

실행(수동 시연):  python src/llm/prescribe.py
환경:  .env 의 OLLAMA_MODEL(기본 qwen2.5:14b) · OLLAMA_HOST(ollama 라이브러리가 자동 참조)
"""
import json
import logging
import os
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

_log = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm.rag import retrieve  # noqa: E402
from llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_forecast  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "너는 토마토 재배 초보자를 돕는 한국어 재배 도우미다.\n"
    "규칙(반드시 지켜라):\n"
    "1) 너는 병을 직접 진단하지 않는다. 사진 진단이 필요하면 반드시 get_diagnosis tool을, "
    "병변 위치가 필요하면 get_detection tool을 호출하고, 그 결과만 근거로 삼는다.\n"
    "2) 네가 아는 진단 범위는 잎곰팡이병·정상·황화잎말이바이러스(tylcv) 3종뿐이다. "
    "그 밖의 병·작물·부위 질문에는 지어내지 말고 '진단 보류'라고 정직하게 답하라.\n"
    "3) tool이 진단을 차단하면(ood_blocked=true) 병명을 절대 단정하지 말고 재촬영을 안내하라.\n"
    "4) 초보자 눈높이로 쉬운 말을 쓰고 어려운 용어는 풀어 설명하라.\n"
    "5) 재배가이드 근거가 제공되면 그 내용에 부합하게 처방하고, 근거에 없는 구체적 약제명·수치는 지어내지 말라.\n"
    "6) 환경 예측(다음날 온도·습도위험)이 제공되면 진단과 교차해 시간축 선제 조치를 제안하라.\n"
    "최종 답변은 지정된 JSON 스키마로만 출력한다."
)


class Prescription(BaseModel):
    """자연어 처방 고정 스키마 (phase3_llm.md §5-5). Streamlit·디스코드가 같은 소스로 사용.

    필드 설명은 model_json_schema() 에 실려 모델의 각 칸 작성을 안내한다(빈약한 답 방지).
    """
    진단요약: str = Field(description="진단 결과를 한 문장으로. 예: '잎곰팡이병으로 보입니다(신뢰도 87%)'")
    원인: str = Field(description="이 병이 왜 생기는지 초보자도 알기 쉽게 설명한 완전한 문장")
    즉시조치: str = Field(description="지금 당장 해야 할 구체적 조치")
    예방: str = Field(description="앞으로 재발을 막기 위한 관리 방법")
    재촬영시점: str = Field(description="언제 다시 사진을 찍어 확인하면 좋은지, 또는 재촬영 방법")
    근거출처: list[str] = Field(default_factory=list,
                             description="처방 근거 출처(RAG 검색 결과로 코드가 채우므로 모델은 비워도 됨)")


def _guard_directive(diag: dict | None) -> str:
    """환각 방어 ①·② — 진단 결과에 따라 최종 답변 톤을 지시."""
    if diag is None:
        return ("진단 도구를 호출하지 않았다. 잎 병해 3종 범위 밖이면 '진단 보류'로 정직하게 답하고, "
                "일반 재배 조언만 신중히 제공하라.")
    if diag.get("error"):
        return ("진단 도구 실행에 실패했다(진단 자체가 수행되지 못함). 병명을 절대 단정하지 말고, "
                "사진을 다시 확인해 재시도하도록 안내하라.")
    if diag.get("ood_blocked"):
        return (f"진단이 차단되었다(사유: {diag.get('reason')}). 병명을 절대 단정하지 말라. "
                "왜 진단할 수 없는지 쉽게 설명하고, '재촬영시점'에 잎 뒷면을 밝은 곳에서 다시 찍는 방법을 담아라.")
    prob = diag.get("prob", 0.0)
    if prob >= 0.8:
        tone = "확신이 높다. 병명을 단정하고 즉시 방제를 안내하라."
    elif prob >= 0.6:
        tone = "확신이 중간이다. 단정하지 말고 '가능성'으로 표현하며 관찰을 함께 권하라."
    else:
        tone = "확신이 낮다. 절대 단정하지 말고 정밀 확인(재촬영·전문가 상담)을 우선 권하라."
    return f"진단 신뢰도 {prob:.0%} — {tone} 아는 진단 범위는 잎 병해 3종뿐임을 잊지 마라."


def _rag_directive(chunks: list[dict]) -> str:
    """RAG — 검색된 재배가이드 근거를 처방의 사실 기반으로 주입."""
    body = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
    return ("아래는 신뢰할 수 있는 재배가이드 근거다. 처방은 이 근거에 부합하게 작성하고, "
            "근거에 없는 구체적 약제명·수치는 지어내지 말라.\n\n" + body)


def _forecast_directive(fc: dict) -> str:
    """시간축 처방(§5-4) — LSTM 환경예측을 진단과 교차해 선제 조치 유도."""
    return (f"환경 예측(LSTM): 다음날 내부온도 {fc['next_temp']}℃({fc['trend']}), "
            f"최근 습도 평균 {fc['humidity_mean']}% → 습도위험 '{fc['humidity_risk']}'. "
            "잎곰팡이병은 고습·야간 결로에서 급속히 번진다. 습도위험이 '높음'·'보통'이면 "
            "야간 환기·제습 같은 환경 선제 조치를 '즉시조치' 또는 '예방'에 시간축으로 포함하라.")


def _rag_sources(chunks: list[dict]) -> list[str]:
    """검색 chunk → 근거출처 문자열 목록(제목·기관명·URL, 중복 제거). 코드가 직접 채워 환각 배제."""
    out: list[str] = []
    for c in chunks:
        label = c.get("title", "")
        if c.get("source_name"):
            label += f" ({c['source_name']})"
        if c.get("source"):
            label += f" — {c['source']}"
        if label and label not in out:
            out.append(label)
    return out


def prescribe(user_msg: str, image_path: str | None = None) -> Prescription:
    """사용자 메시지(+선택 잎 사진) → 구조화 자연어 처방.

    LLM이 tool을 호출하면 실제 추론을 실행해 결과를 되먹이고(최대 MAX_TOOL_ROUNDS),
    진단 신뢰도/차단 여부에 따라 톤 지시문을 주입한 뒤 JSON 스키마로 최종 처방을 받는다.
    """
    content = user_msg
    if image_path:
        content += f"\n\n(첨부된 잎 사진 경로: {image_path})"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]

    diag = None
    for _ in range(MAX_TOOL_ROUNDS):
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        for tc in calls:
            name = tc["function"]["name"]
            args = dict(tc["function"]["arguments"] or {})
            # 신뢰 경로 강제: LLM이 베낀 경로 대신 호출자가 준 image_path 사용
            # → 전사 오류·임의 경로 주입 동시 차단.
            if image_path and "image_path" in args:
                args["image_path"] = image_path
            fn = TOOL_REGISTRY.get(name)
            if fn is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:                   # tool 실패도 모델에 알려줌(환각 방지)
                    result = {"error": f"{type(e).__name__}: {e}"}
            if name == "get_diagnosis":
                diag = result
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, ensure_ascii=False)})
    else:
        _log.warning("tool 호출이 %d라운드 내 종료되지 않음 — 마지막 상태로 처방 생성.", MAX_TOOL_ROUNDS)

    messages.append({"role": "system", "content": _guard_directive(diag)})

    # RAG — 진단됐고 차단·오류 아니면 라벨로 재배가이드 근거 검색 후 주입
    rag_chunks: list[dict] = []
    if diag and diag.get("label") and not diag.get("ood_blocked") and not diag.get("error"):
        try:
            rag_chunks = retrieve(user_msg, disease=diag["label"], k=3)
        except Exception as e:                           # 임베딩/코퍼스 실패해도 처방은 계속
            _log.warning("RAG 검색 실패 — 근거 없이 진행: %s", e)
        if rag_chunks:
            messages.append({"role": "system", "content": _rag_directive(rag_chunks)})
    sources = _rag_sources(rag_chunks)

    # 시간축 처방 — 습도 민감 병해(잎곰팡이병)면 환경 예측을 교차 주입
    if diag and diag.get("label") == "leaf_mold" and not diag.get("ood_blocked") and not diag.get("error"):
        fc = get_forecast()
        if fc and not fc.get("unavailable"):
            messages.append({"role": "system", "content": _forecast_directive(fc)})

    last_err = None
    for _ in range(2):                                   # 스키마 위반 시 1회 재시도
        final = ollama.chat(model=MODEL, messages=messages,
                            format=Prescription.model_json_schema())
        try:
            presc = Prescription.model_validate_json(final["message"]["content"])
            presc.근거출처 = sources                       # 근거는 코드가 채움(LLM 환각 배제)
            return presc
        except ValidationError as e:
            last_err = e
            messages.append({"role": "system",
                             "content": "직전 출력이 스키마와 맞지 않았다. 반드시 지정된 JSON 스키마로만 다시 출력하라."})
    _log.warning("구조화 출력 검증 실패 — 안전 폴백 반환: %s", last_err)
    return Prescription(진단요약="처방 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                        원인="-", 즉시조치="-", 예방="-", 재촬영시점="-", 근거출처=sources)


if __name__ == "__main__":
    # 수동 시연 — data/tomato/val/leaf_mold 샘플 1장으로 처방 출력
    sample_dir = ROOT / "data" / "tomato" / "val" / "leaf_mold"
    samples = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png"))
    if not samples:
        print(f"샘플 없음: {sample_dir}")
        sys.exit(1)
    img = str(samples[0])
    print(f"[입력] {img}\n[모델] {MODEL}\n처방 생성 중...\n")
    p = prescribe("이 토마토 잎 사진 좀 봐줘. 병이면 어떻게 조치해야 해?", image_path=img)
    print(json.dumps(p.model_dump(), ensure_ascii=False, indent=2))
