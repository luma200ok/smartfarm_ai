"""Pydantic v2 Request/Response 스키마 — `api/routers/*` 전용.

처방 응답은 `src/llm/prescribe.py`의 `Prescription`(이미 pydantic BaseModel)을 그대로 재사용한다
(핸드오프 확정 — 별도 응답 스키마로 감싸지 않음).
"""
from dl import infer
from pydantic import BaseModel, Field, field_validator


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


class DiagnosisIn(BaseModel):
    """`/api/prescriptions`의 `diagnosis` 폼 필드(JSON 문자열) 검증(code-reviewer P2-1).

    이전엔 `json.loads()`만 하고 dict라 가정해 바로 `prescribe_fast(diag=...)`에 넘겼다 —
    `"5"`·`"[1,2,3]"` 같은 비-dict JSON이 들어오면 내부에서 `.get()` 호출 시 500이 났다
    (재현 케이스는 tests/test_api_prescriptions.py). pydantic이 dict 형태를 강제하고,
    `label`은 `dl.infer.CLASSES` 화이트리스트(+None)만 허용해 임의 라벨 위조도 막는다.
    """

    ood_blocked: bool = False
    reason: str | None = None
    label: str | None = None
    label_kr: str | None = None
    prob: float | None = Field(default=None, ge=0.0, le=1.0)
    probs: dict[str, float] | None = None
    part: str | None = None

    @field_validator("label")
    @classmethod
    def _label_in_whitelist(cls, v: str | None) -> str | None:
        if v is not None and v not in infer.CLASSES:
            raise ValueError(f"알 수 없는 진단 라벨: {v}")
        return v

    @field_validator("probs")
    @classmethod
    def _probs_in_range(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            for k, p in v.items():
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"probs[{k}] 값이 0.0~1.0 범위를 벗어남: {p}")
        return v
