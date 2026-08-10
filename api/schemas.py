"""Pydantic v2 Request/Response 스키마 — `api/routers/*` 전용.

처방 응답은 `src/llm/prescribe.py`의 `Prescription`(이미 pydantic BaseModel)을 그대로 재사용한다
(핸드오프 확정 — 별도 응답 스키마로 감싸지 않음).
"""
from pydantic import BaseModel


class OllamaStatus(BaseModel):
    online: bool
    models: list[str] | None = None


class ArtifactStatus(BaseModel):
    resnet18: bool
    yolo: bool
    part: bool
    leaf_gate: bool


class HealthResponse(BaseModel):
    status: str
    ollama: OllamaStatus
    artifacts: ArtifactStatus


class DetectionBox(BaseModel):
    label: str
    conf: float


class YoloResult(BaseModel):
    annotated_png_base64: str
    boxes: list[DetectionBox]


class DiagnosisResponse(BaseModel):
    """게이트 차단 시(ood_blocked=True)에도 200 — label 이하 필드는 None(현행 뷰 동작과 동일 의미)."""

    ood_blocked: bool
    reason: str | None = None
    label: str | None = None
    label_kr: str | None = None
    prob: float | None = None
    probs: dict[str, float] | None = None
    part: str | None = None
    cam_png_base64: str | None = None
    yolo: YoloResult | None = None
