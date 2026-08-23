"""
smartfarm_ai#84 — RAG 자유 질의 챗(POST /api/chat) 전용 오케스트레이션.

`prescribe.py`(사진 진단 + 구조화 처방, tool calling)와 달리 이미지·진단 결과 없이 순수 텍스트
Q&A만 다룬다 — tool 호출·JSON 스키마 강제 없이 RAG 근거를 주입한 뒤 자유 텍스트 답변 1회만
받는다(1-call, prescribe_fast와 동일한 "무거운 오케스트레이션 없이 가볍게" 원칙).

환각 방어: ① 스코프 한정(system 프롬프트에 토마토 병해·재배 환경으로 답변 범위 고정)
          ② RAG 근거 있으면 그에 부합하게, 없으면 지어내지 말라는 지시
          ③ sources는 코드가 RAG 검색 결과로 직접 채움(LLM이 채우지 않음, prescribe.py와 동일 원칙)

모델: `OLLAMA_MODEL` env를 그대로 재사용한다(신규 env 불필요) — prescribe.py의 agentic 경로와
같은 모델을 쓰며, 배포 서버는 이미 `OLLAMA_MODEL=qwen2.5:7b`로 설정돼 있다(docs/STATUS.md).
"""
import logging
import os
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import history  # noqa: E402
from llm.rag import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

# prescribe.py의 _write_final과 동일한 타임아웃 계약(무한 대기만 차단, 정상 지연은 여유 있게 통과).
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
KEEP_ALIVE = "30m"

_FALLBACK_ANSWER = "죄송해요, 지금은 답변을 드리기 어려워요. 잠시 후 다시 시도해 주세요."

_client_instance: "ollama.Client | None" = None


def _client() -> "ollama.Client":
    """지연 생성 싱글턴(프로세스 1회). 테스트 seam: `monkeypatch.setattr(chat, "_client", ...)`."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ollama.Client(timeout=OLLAMA_TIMEOUT)
    return _client_instance


SYSTEM_PROMPT = (
    "너는 토마토 재배를 돕는 한국어 농업 도우미다.\n"
    "규칙(반드시 지켜라):\n"
    "1) 답변 범위는 토마토 병해충·재배 환경(온습도·관수·시비 등) 관련 질문으로 한정한다.\n"
    "2) 아래에 재배가이드 근거가 주어지면 그에 부합하게 답하고, 근거에 없는 약제명·수치는 "
    "지어내지 않는다.\n"
    "3) 범위를 벗어난 질문(재배와 무관한 잡담·다른 작물 등)에는 자세히 답하지 말고, "
    "'저는 토마토 병해·재배 환경 질문을 도와드려요' 같은 짧은 안내만 한다.\n"
    "4) 초보자도 이해할 수 있는 쉬운 말로 간결하게 답한다.\n"
    "5) 잎 사진 진단이 필요한 질문이면 사진을 올려 진단·처방을 받아보라고 안내한다."
)


class ChatAnswer(BaseModel):
    """`POST /api/chat` 응답 스키마(신규 영문 필드 — 기존 Prescription 한글 스키마와 별개,
    docs/api-contract.md §4.7). LLM 실패 시에도 200 + `fallback=true`로 안전 폴백한다."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    fallback: bool = False


def _rag_directive(chunks: list[dict]) -> str:
    """RAG — 검색된 재배가이드 근거를 답변의 사실 기반으로 주입(prescribe.py와 동일 원칙)."""
    body = "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)
    return ("아래는 신뢰할 수 있는 재배가이드 근거다. 이에 부합하게 답하고, 근거에 없는 "
            "약제명·수치는 지어내지 말라.\n\n" + body)


def _rag_sources(chunks: list[dict]) -> list[str]:
    """검색 chunk → 출처 문자열 목록(제목·기관명·URL, 중복 제거). 코드가 직접 채워 환각 배제."""
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


def answer_chat(question: str, caller_ref: str | None = None) -> ChatAnswer:
    """자유 질의 1건 → RAG 근거 주입 답변. LLM 예외(연결 실패·타임아웃 등) 시 200 + 안내문 +
    fallback=True로 안전 폴백한다(처방과 동일 트레이드오프). 이력은 성공·폴백 모두 저장 시도한다.

    caller_ref(smartfarm_ai#66과 동일 컨벤션) — 이력 태깅용 optional 식별자. 저장에만 전달한다.
    """
    rag_chunks: list[dict] = []
    try:
        rag_chunks = retrieve(question, disease=None, k=3)
    except Exception as e:                                # RAG 실패해도 답변은 계속 진행
        _log.warning("챗 RAG 검색 실패 — 근거 없이 진행: %s", e)
    sources = _rag_sources(rag_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if rag_chunks:
        messages.append({"role": "system", "content": _rag_directive(rag_chunks)})
    messages.append({"role": "user", "content": question})

    try:
        resp = _client().chat(model=MODEL, messages=messages, keep_alive=KEEP_ALIVE)
        answer_text = (resp["message"]["content"] or "").strip()
        if not answer_text:
            raise ValueError("LLM이 빈 응답을 반환함")
        result = ChatAnswer(answer=answer_text, sources=sources, fallback=False)
    except Exception as e:
        _log.warning("챗 LLM 호출 실패 — 안전 폴백 반환: %s", e)
        result = ChatAnswer(answer=_FALLBACK_ANSWER, sources=sources, fallback=True)

    history.save_chat(question, result.answer, result.sources, caller_ref=caller_ref)
    return result
