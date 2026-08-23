"""POST /api/chat — RAG 자유 질의(병해·재배 환경 중심) + 이력 저장(smartfarm_ai#84).

이미지·진단 결과 없이 순수 텍스트 Q&A만 다룬다 — 기존 prescriptions 라우터의 Form 파싱·
ApiError·동시성 캡 컨벤션을 그대로 따른다(docs/api-contract.md §4.7). 프롬프트·LLM 호출·안전
폴백은 `src/llm/chat.py`(처방 전용 prescribe.py와 별도 모듈)가 맡고, 이 라우터는 폼 파싱과
동시성 슬롯 확보만 담당한다(비즈니스 로직 금지 컨벤션).
"""
from fastapi import APIRouter, Form
from llm.chat import ChatAnswer, answer_chat

from ..concurrency import chat_slot, inference_slot

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatAnswer)
def create_chat(
    question: str = Form(..., min_length=1, max_length=500),
    caller_ref: str | None = Form(
        None, max_length=64,
        description="이력 테넌시 태깅용 optional 호출자 식별자(smartfarm_ai#66과 동일 컨벤션, 최대 64자)",
    ),
) -> ChatAnswer:
    # chat_slot() 먼저(챗 하위 상한 1, security-reviewer P2 smartfarm_ai#84) → 그 안에서
    # 공유 inference_slot()(진단·처방과 합계 2 공유). 둘 다 non-blocking·즉시 429, 해제는
    # with 중첩 종료 시 자동 역순이라 데드락 없음.
    with chat_slot():
        with inference_slot():
            return answer_chat(question, caller_ref=caller_ref)
