"""요청 동시성 캡(code-reviewer P2-2) — 3코어 CPU 서버에서 무거운 추론 요청(진단·처방)이
동시에 쌓여 서버 전체가 먹통 되는 자체 DoS를 막는다. `diagnoses`·`prescriptions`에만 적용하고
`health`는 가볍고 상태 확인 용도라 캡 대상에서 제외한다.

두 엔드포인트가 슬롯을 공유(모듈 수준 세마포어 1개)한다 — 진단·처방 모두 같은 3코어를 두고
경쟁하므로, 엔드포인트별로 따로 캡을 두면(2+2) 합계 4개 동시 추론까지 허용해버려 취지가 흐려진다.
"""
import threading
from contextlib import contextmanager
from typing import Iterator

from .errors import ApiError

MAX_CONCURRENT_INFERENCE = 2
_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)


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
