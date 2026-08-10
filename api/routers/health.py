"""GET /api/health — Ollama 온라인 여부(+설치 모델) · 모델 아티팩트(체크포인트 파일) 존재 여부.

Ollama 오프라인이어도 200(서비스 자체는 정상, LLM 기능만 unavailable) — 현행
`app/views/prescribe.py:_ollama_online()` 로직을 그대로 이식(비즈니스 로직 변경 없음).

health는 `api/concurrency.py`의 동시성 캡 대상이 아니다(가볍고 상태 확인 용도) — 대신
Ollama 데몬이 응답 없이 멈춰 있어도 헬스체크 자체가 hang 하지 않도록 짧은 타임아웃(5s)의
전용 클라이언트를 쓴다(P2-4).
"""
from dl import infer
from fastapi import APIRouter

from ..schemas import ArtifactStatus, HealthResponse, OllamaStatus

router = APIRouter(tags=["health"])

_HEALTH_OLLAMA_TIMEOUT = 5.0
_health_ollama_client = None


def _ollama_health_client():
    """헬스체크 전용 Ollama 클라이언트(지연 생성, 5s 타임아웃) — 프로세스 1회만 생성.

    테스트 seam: `monkeypatch.setattr(health, "_ollama_health_client", lambda: fake)`.
    """
    global _health_ollama_client
    if _health_ollama_client is None:
        import ollama

        _health_ollama_client = ollama.Client(timeout=_HEALTH_OLLAMA_TIMEOUT)
    return _health_ollama_client


def _ollama_status() -> OllamaStatus:
    """`ollama.list()` 성공 시 온라인 + 설치 모델명 목록, 실패(데몬 미기동·타임아웃 등) 시 오프라인."""
    try:
        resp = _ollama_health_client().list()
        return OllamaStatus(online=True, models=[m.model for m in resp.models])
    except Exception:
        return OllamaStatus(online=False, models=None)


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama=_ollama_status(),
        artifacts=ArtifactStatus(
            resnet18=infer.CKPT.exists(),
            yolo=infer.YOLO_CKPT.exists(),
            part=infer.PART_CKPT.exists(),
            # leaf_gate(OOD 게이트)는 로컬 체크포인트 파일이 아니라 torchvision 사전학습
            # 가중치(ResNet18_Weights.IMAGENET1K_V1)로 동작 — 파일 유무 대신 상시 가용으로 표기.
            leaf_gate=True,
        ),
    )
