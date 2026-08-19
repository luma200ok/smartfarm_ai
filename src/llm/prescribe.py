"""
Phase 3-1 — Ollama(qwen2.5:14b) 자연어 처방 오케스트레이션.

LLM 주도 function calling: LLM이 get_diagnosis/get_detection tool을 스스로 호출 →
우리가 실제 DL 추론(src/llm/tools.py) 실행 → 결과를 받아 LLM이 처방 생성.

환각 방어 3종:
  ① 신뢰도 톤 분기   — 진단 prob 밴드(>0.8 단정 / 0.6~0.8 중립 / <0.6 정밀확인)로 지시문 주입
  ② 게이트 차단 안내 — ood_blocked면 병명 단정 금지, 재촬영 안내
  ③ 클래스 한정성    — 아는 범위(잎 병해 4종)를 system 프롬프트에 고정, 밖이면 '진단 보류'
최종 출력은 Prescription JSON 스키마로 강제(format=).

실행(수동 시연):  python src/llm/prescribe.py
환경:  .env 의 OLLAMA_MODEL(기본 qwen2.5:14b) · OLLAMA_HOST(ollama 라이브러리가 자동 참조)
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

_log = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import history  # noqa: E402
from llm.rag import retrieve  # noqa: E402
from llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_diagnosis, get_forecast  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
# 이슈 #18 — fast-path 최종 처방 작성 전용 모델(서버 벤치: exaone3.5:2.4b가 품질·속도 우수).
# 미설정 시 기존 MODEL로 폴백(하위호환).
WRITER_MODEL = os.getenv("OLLAMA_WRITER_MODEL") or MODEL

# API 리뷰 P2-4 — 최종 처방 JSON 생성 호출(_write_final, fast-path·agentic 공유)에 타임아웃을
# 건다. module-level ollama.chat(기본 무제한)은 Ollama가 응답을 영영 안 주면 요청이 그대로
# 행(hang)한다 — 서버 cold 경로(모델 로드 대기)를 감안해 기본 180s로 보수적 설정(무한 대기만
# 차단하는 목적, 정상 응답 지연은 여유 있게 통과). agentic prescribe()의 tool 라운드 호출은
# 건드리지 않음(리뷰 스코프 — 공유 헬퍼 경로에만 적용).
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
_client_instance: "ollama.Client | None" = None


def _client() -> "ollama.Client":
    """`_write_final` 전용 타임아웃 클라이언트(지연 생성, 프로세스 1회).

    테스트 seam: `monkeypatch.setattr(prescribe, "_client", lambda: fake)`로 이 함수를
    통째로 갈아끼운다(module-level `ollama.chat`을 그대로 쓰는 tool 라운드 호출은
    기존처럼 `patch("ollama.chat", ...)`로 모킹).
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = ollama.Client(timeout=OLLAMA_TIMEOUT)
    return _client_instance


MAX_TOOL_ROUNDS = 4
KEEP_ALIVE = "30m"
# tool 라운드 중 자유텍스트로 흐를 때(다음 라운드에서 버려지는 응답) 낭비를 줄이는 캡.
# 128은 tool_calls 자체까지 잘라 조용한 진단 미실행("진단 보류")을 유발할 수 있어 512로 상향—
# 여전히 300tok대 낭비 라운드보다는 짧지만, 정상 tool_calls 토큰 길이엔 여유를 둔다.
TOOL_ROUND_NUM_PREDICT = 512

SYSTEM_PROMPT = (
    "너는 토마토 재배 초보자를 돕는 한국어 재배 도우미다.\n"
    "규칙(반드시 지켜라):\n"
    "1) 병은 직접 진단하지 않는다. 사진 진단은 get_diagnosis, 병변 위치는 get_detection tool을 호출해 "
    "그 결과만 근거로 삼는다.\n"
    "2) 아는 범위는 잎마름역병(late_blight)·잎곰팡이병·정상·tylcv 4종뿐 — 그 밖은 지어내지 말고 '진단 보류'.\n"
    "3) ood_blocked=true면 병명을 단정하지 말고 재촬영을 안내한다.\n"
    "4) 초보자 눈높이 쉬운 말로, 어려운 용어는 풀어 설명한다.\n"
    "5) 재배가이드 근거가 있으면 그에 부합하게 처방하고, 근거 없는 약제명·수치는 지어내지 않는다.\n"
    "6) 환경 예측(다음날 온도·습도위험)이 있으면 진단과 교차해 시간축 선제 조치를 제안한다.\n"
    "7) 날씨 질문이면 get_weather tool을 호출해 토마토(작물) 관점으로 해석해 답한다.\n"
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
    fallback: bool = Field(default=False,
                            description="스키마 위반 재시도(1회) 후에도 실패해 안전 폴백 문구를 반환했으면 True"
                                         "(smartfarm_ai#66) — 서비스가 클라이언트에 저품질 응답을 구분해 표시할 때 사용")


def _llm_response_schema() -> dict:
    """LLM에 요청할 JSON 스키마 — `fallback`은 서버가 최종 강제하는 전용 필드라(근거출처와 동일
    원칙) 모델 요청 스키마에서 제외한다(P1 픽스, smartfarm_ai#66). 그대로 두면 LLM이 임의로
    `"fallback": true`를 채워도 검증을 통과해 오염된 값이 응답에 그대로 실릴 수 있었다."""
    schema = Prescription.model_json_schema()
    schema["properties"].pop("fallback", None)
    if "required" in schema:
        schema["required"] = [f for f in schema["required"] if f != "fallback"]
    return schema


def _guard_directive(diag: dict | None) -> str:
    """환각 방어 ①·② — 진단 결과에 따라 최종 답변 톤을 지시."""
    if diag is None:
        return "진단 도구를 호출하지 않았다. 범위 밖이면 '진단 보류'로 정직하게 답하고, 일반 조언만 제공하라."
    if diag.get("error"):
        return "진단 도구 실행 실패. 병명을 절대 단정하지 말고, 사진을 다시 확인해 재시도하도록 안내하라."
    if diag.get("ood_blocked"):
        return (f"진단 차단(사유: {diag.get('reason')}). 병명을 절대 단정하지 말라. 이유를 쉽게 설명하고 "
                "'재촬영시점'에 잎 뒷면을 밝은 곳에서 다시 찍는 방법을 담아라.")
    prob = diag.get("prob", 0.0)
    if prob >= 0.8:
        tone = "확신이 높다. 병명을 단정하고 즉시 방제를 안내하라."
    elif prob >= 0.6:
        tone = "확신이 중간이다. 단정하지 말고 '가능성'으로 표현하며 관찰을 권하라."
    else:
        tone = "확신이 낮다. 절대 단정하지 말고 정밀 확인(재촬영·전문가 상담)을 우선 권하라."
    return f"진단 신뢰도 {prob:.0%} — {tone} 아는 범위는 잎 병해 4종뿐임을 잊지 마라."


def _rag_directive(chunks: list[dict]) -> str:
    """RAG — 검색된 재배가이드 근거를 처방의 사실 기반으로 주입."""
    body = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
    return ("아래는 신뢰할 수 있는 재배가이드 근거다. 이에 부합하게 처방하고, 근거에 없는 "
            "약제명·수치는 지어내지 말라.\n\n" + body)


def _forecast_directive(fc: dict) -> str:
    """시간축 처방(§5-4) — LSTM 환경예측을 진단과 교차해 선제 조치 유도."""
    return (f"환경 예측(LSTM): 다음날 내부온도 {fc['next_temp']}℃({fc['trend']}), "
            f"최근 습도 평균 {fc['humidity_mean']}% → 습도위험 '{fc['humidity_risk']}'. "
            "잎곰팡이병은 고습·결로에서 급속히 번진다. 습도위험이 '높음'·'보통'이면 "
            "야간 환기·제습을 '즉시조치' 또는 '예방'에 포함하라.")


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


WRITER_SYSTEM_PROMPT = (
    "너는 토마토 재배 초보자를 돕는 한국어 재배 도우미다. 아래 진단·근거·예측 정보를 바탕으로 "
    "최종 처방만 작성한다(도구 호출은 하지 않는다).\n"
    "규칙(반드시 지켜라):\n"
    "1) 아는 범위는 잎마름역병(late_blight)·잎곰팡이병·정상·tylcv 4종뿐 — 그 밖은 지어내지 말고 '진단 보류'.\n"
    "2) 진단이 차단(ood_blocked)됐으면 병명을 단정하지 말고 재촬영을 안내한다.\n"
    "3) 초보자 눈높이 쉬운 말로, 어려운 용어는 풀어 설명한다.\n"
    "4) 재배가이드 근거가 있으면 그에 부합하게 처방하고, 근거 없는 약제명·수치는 지어내지 않는다.\n"
    "5) 환경 예측(다음날 온도·습도위험) 정보가 있으면 진단과 교차해 시간축 선제 조치를 제안한다.\n"
    "최종 답변은 지정된 JSON 스키마로만 출력한다."
)


def _inject_directives(messages: list, user_msg: str, diag: dict | None,
                        on_progress) -> list[str]:
    """공통 후처리 — guard·RAG·forecast 지시문을 messages에 주입하고 근거출처 목록을 돌려준다.

    prescribe()(agentic)·prescribe_fast()(#18) 양쪽이 공유한다.
    """
    messages.append({"role": "system", "content": _guard_directive(diag)})

    rag_chunks: list[dict] = []
    if diag and diag.get("label") and not diag.get("ood_blocked") and not diag.get("error"):
        on_progress("rag")
        try:
            rag_chunks = retrieve(user_msg, disease=diag["label"], k=3)
        except Exception as e:                           # 임베딩/코퍼스 실패해도 처방은 계속
            _log.warning("RAG 검색 실패 — 근거 없이 진행: %s", e)
        if rag_chunks:
            messages.append({"role": "system", "content": _rag_directive(rag_chunks)})
    sources = _rag_sources(rag_chunks)

    if diag and diag.get("label") in ("leaf_mold", "late_blight") and not diag.get("ood_blocked") and not diag.get("error"):
        on_progress("forecast")
        fc = get_forecast()
        if fc and not fc.get("unavailable"):
            messages.append({"role": "system", "content": _forecast_directive(fc)})

    return sources


def _write_final(messages: list, user_msg: str, image_path: str | None, diag: dict | None,
                  sources: list[str], on_progress, model: str,
                  caller_ref: str | None = None) -> Prescription:
    """공통 후처리 — 최종 구조화 JSON 생성(스키마 위반 1회 재시도) + 안전 폴백 + history 저장.

    caller_ref(smartfarm_ai#66) — 이력 테넌시 태깅용 optional 식별자. 그대로
    `history.save_prescription`에 전달만 하고 처방 생성 로직에는 관여하지 않는다.
    """
    on_progress("writing")
    last_err = None
    for _ in range(2):                                   # 스키마 위반 시 1회 재시도
        final = _client().chat(model=model, messages=messages,
                               format=_llm_response_schema(), keep_alive=KEEP_ALIVE)
        try:
            presc = Prescription.model_validate_json(final["message"]["content"])
            presc.근거출처 = sources                       # 근거는 코드가 채움(LLM 환각 배제)
            presc.fallback = False                        # 정상 경로 강제(P1 픽스) — LLM이
            # 요청 스키마에서 제외됐어도 프롬프트 텍스트로 흉내내 채워보낸 값을 신뢰하지 않는다.
            history.save_prescription(user_msg, image_path, diag, presc, caller_ref=caller_ref)
            return presc
        except ValidationError as e:
            last_err = e
            messages.append({"role": "system",
                             "content": "직전 출력이 스키마와 맞지 않았다. 반드시 지정된 JSON 스키마로만 다시 출력하라."})
    _log.warning("구조화 출력 검증 실패 — 안전 폴백 반환: %s", last_err)
    presc = Prescription(진단요약="처방 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                        원인="-", 즉시조치="-", 예방="-", 재촬영시점="-", 근거출처=sources,
                        fallback=True)
    history.save_prescription(user_msg, image_path, diag, presc, caller_ref=caller_ref)
    return presc


def prescribe_fast(user_msg: str, image_path: str | None = None,
                    on_progress: "Callable[[str], None] | None" = None,
                    diag: dict | None = None,
                    caller_ref: str | None = None) -> Prescription:
    """이슈 #18 — fast-path(1-call) 처방. tool 실행을 코드가 직행시키고 LLM은 최종 작성 1회만.

    처방 버튼 경로 전용(흐름 고정). CLI·비교 데모는 기존 prescribe()(agentic tool calling)를 유지.

    diag(#18 P2 픽스) — 호출자가 표시용으로 이미 get_diagnosis를 실행해뒀다면 그 결과를 넘겨
    재사용한다(DL 추론 2회 실행 방지). None이고 image_path가 있으면 기존대로 직접 호출한다.

    caller_ref(smartfarm_ai#66) — optional 이력 태깅 식별자. history 저장에만 전달한다.
    """
    def _progress(stage: str) -> None:
        if on_progress is None:
            return
        try:                                              # 콜백 예외가 처방 자체를 죽이지 않게
            on_progress(stage)
        except Exception as e:
            _log.warning("on_progress 콜백 실패(%s) — 처방은 계속 진행: %s", stage, e)

    if diag is not None:
        _progress("diagnosis")                            # 이미 완료된 단계 — 즉시 완료 표시만
    elif image_path:
        _progress("diagnosis")
        try:
            diag = get_diagnosis(image_path)
        except Exception as e:                            # tool 실패도 정직하게 알려줌(환각 방지)
            diag = {"error": f"{type(e).__name__}: {e}"}

    content = user_msg
    if image_path:
        content += f"\n\n(첨부된 잎 사진 경로: {image_path})"
    messages = [{"role": "system", "content": WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    if diag is not None:
        messages.append({"role": "system",
                         "content": "진단 결과: " + json.dumps(diag, ensure_ascii=False)})

    sources = _inject_directives(messages, user_msg, diag, _progress)
    return _write_final(messages, user_msg, image_path, diag, sources, _progress, WRITER_MODEL,
                        caller_ref=caller_ref)


def prescribe(user_msg: str, image_path: str | None = None,
              on_progress: "Callable[[str], None] | None" = None) -> Prescription:
    """사용자 메시지(+선택 잎 사진) → 구조화 자연어 처방.

    LLM이 tool을 호출하면 실제 추론을 실행해 결과를 되먹이고(최대 MAX_TOOL_ROUNDS),
    진단 신뢰도/차단 여부에 따라 톤 지시문을 주입한 뒤 JSON 스키마로 최종 처방을 받는다.

    on_progress(#15 C3) — 단계 전환 시점(진단 실행/근거 검색/예보 확인/처방 작성)에 호출되는
    선택적 콜백. 기본 None이면 기존과 완전히 동일하게 동작한다(하위호환, CLI·pipeline.py 무영향).
    """
    def _progress(stage: str) -> None:
        if on_progress is None:
            return
        try:                                              # P2-1: 콜백 예외가 처방 자체를 죽이지 않게
            on_progress(stage)
        except Exception as e:
            _log.warning("on_progress 콜백 실패(%s) — 처방은 계속 진행: %s", stage, e)

    content = user_msg
    if image_path:
        content += f"\n\n(첨부된 잎 사진 경로: {image_path})"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]

    diag = None
    _progress("diagnosis")
    for _ in range(MAX_TOOL_ROUNDS):
        # C1(#15) — 다음 tool 라운드에서 어차피 버려지는 자유텍스트 장문 생성을 캡으로 축소.
        # tool_calls를 원하면 여전히 허용(tools=는 그대로 넘김) — 환각방어·MAX_TOOL_ROUNDS 로직 불변.
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
                           options={"num_predict": TOOL_ROUND_NUM_PREDICT}, keep_alive=KEEP_ALIVE)
        msg = resp["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            if image_path and diag is None:
                # 이미지가 첨부됐는데 get_diagnosis가 한 번도 호출되지 않은 채 종료 —
                # num_predict 캡에 의한 tool_calls 잘림 의심을 포함해 가시화(조용한 "진단 보류" 방지).
                _log.warning(
                    "이미지가 첨부됐는데 tool 호출 없이 종료됨(num_predict=%d 캡에 의한 잘림 의심) "
                    "— 진단 미실행 상태로 처방 생성됨.", TOOL_ROUND_NUM_PREDICT)
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

    sources = _inject_directives(messages, user_msg, diag, _progress)
    return _write_final(messages, user_msg, image_path, diag, sources, _progress, MODEL)


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
