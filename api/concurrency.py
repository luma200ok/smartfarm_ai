"""요청 동시성 캡(code-reviewer P2-2) — 3코어 CPU 서버에서 무거운 추론 요청(진단·처방·챗)이
동시에 쌓여 서버 전체가 먹통 되는 자체 DoS를 막는다. `diagnoses`·`prescriptions`·`chat`에만
적용하고 `health`는 가볍고 상태 확인 용도라 캡 대상에서 제외한다.

세 엔드포인트가 슬롯을 공유(모듈 수준 세마포어 1개, `inference_slot()`)한다 — 모두 같은 3코어를
두고 경쟁하므로, 엔드포인트별로 따로 캡을 두면 합계가 늘어나 취지가 흐려진다.

**챗 하위 상한(P2, security-reviewer, smartfarm_ai#84)** — 챗은 이미지 업로드도 진단 게이트도
없이 텍스트 500자만으로 익명 방문자가 호출 가능해, 공유 슬롯 2개를 챗이 전부 점유하면
`OLLAMA_TIMEOUT`(기본 180s) 동안 진단·처방이 굶는다(가용성 저하). `MAX_CONCURRENT_CHAT = 1`
+ 별도 세마포어(`chat_slot()`)를 챗 라우트가 `inference_slot()`보다 먼저 획득하게 해, 챗이
공유 슬롯 중 최대 1개만 쓰도록 상한을 건다 — 진단·처방 전용으로 최소 1슬롯이 항상 남는다.
챗 라우트는 `with chat_slot(): with inference_slot(): ...`(중첩) 순서로 획득하고, 해제는
with 블록 종료 시 자동으로 역순(inference_slot 먼저, chat_slot 나중)이라 데드락 걱정이 없다.
"""
import threading
from contextlib import contextmanager
from typing import Iterator

from .errors import ApiError

MAX_CONCURRENT_INFERENCE = 2
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)

MAX_CONCURRENT_CHAT = 1
_chat_slots = threading.BoundedSemaphore(MAX_CONCURRENT_CHAT)


@contextmanager
def inference_slot() -> Iterator[None]:
    """슬롯 확보 실패(이미 꽉 참) 시 즉시 429 — 대기(blocking)하지 않는다(요청 적체 방지)."""
    acquired = _inference_slots.acquire(blocking=False)
    if not acquired:
        raise ApiError(429, "서버가 혼잡합니다. 잠시 후 다시 시도하세요.")
    try:
        yield
    finally:
        _inference_slots.release()


@contextmanager
def chat_slot() -> Iterator[None]:
    """챗 전용 하위 상한(smartfarm_ai#84 P2) — `inference_slot()`보다 먼저 획득해야 한다.

    이 슬롯을 먼저 확보한 요청만 공유 `inference_slot()`을 시도하므로, 챗이 동시에 쓸 수 있는
    공유 슬롯은 `MAX_CONCURRENT_CHAT`(1)개로 제한된다. 확보 실패 시 즉시 429(non-blocking,
    `inference_slot()`과 동일 계약).
    """
    acquired = _chat_slots.acquire(blocking=False)
    if not acquired:
        raise ApiError(429, "서버가 혼잡합니다. 잠시 후 다시 시도하세요.")
    try:
        yield
    finally:
        _chat_slots.release()
