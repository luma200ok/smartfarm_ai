"""POST /api/prescriptions — 질문(+선택 이미지 또는 기존 진단 재사용) → LLM 자연어 처방.

`llm.prescribe.prescribe_fast(question, image_path, diag)` 인터페이스를 그대로 따른다(fast-path
1-call, 서버 웜 ~16s). 이미지·진단 모두 없어도 동작(일반 질문 → DL 진단 없이 처방).

요청은 multipart/form-data로 받는다 — `file`(선택, 이미지) 하나의 엔드포인트에서 다루려면
FastAPI에서 JSON body와 업로드 파일을 함께 받을 수 없어(순수 JSON이면 파일 필드 자체가 불가),
`diagnosis`는 JSON 문자열을 담는 Form 필드로 표현한다(핸드오프의 "JSON {question, diagnosis?}"
의도는 유지하되 전송 형식만 multipart로 통일 — 이미지 없는 클라이언트도 동일 엔드포인트로 호출
가능해 분기 부담이 없다).

이미지 전달: UploadFile → (사전 검증 후) tempfile 저장 → 경로 전달(`llm.tools.get_diagnosis`가
경로 기반이라 `prescribe_fast`도 경로를 받음) — 응답 후 임시파일 정리(finally). 무거운 처리
(DL 진단 + LLM 호출)는 `api/concurrency.py`의 동시성 캡(최대 2) 안에서만 실행한다.
"""
import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from llm.prescribe import Prescription, prescribe_fast
from pydantic import ValidationError

from ..concurrency import inference_slot
from ..errors import ApiError
from ..schemas import DiagnosisIn
from ..uploads import load_validated_bytes

_log = logging.getLogger(__name__)

router = APIRouter(tags=["prescriptions"])


def _parse_diagnosis(diagnosis: str | None) -> dict | None:
    """`diagnosis` Form 필드(JSON 문자열) → 검증된 dict(P2-1). 실패 시 422(상세는 로그로만)."""
    if not diagnosis:
        return None
    try:
        parsed = json.loads(diagnosis)
    except json.JSONDecodeError as e:
        _log.warning("diagnosis 필드 JSON 파싱 실패: %s", e)
        raise ApiError(422, "diagnosis 필드가 유효한 JSON이 아닙니다.") from e
    try:
        return DiagnosisIn.model_validate(parsed).model_dump(exclude_none=True)
    except ValidationError as e:
        _log.warning("diagnosis 필드 검증 실패: %s", e)
        raise ApiError(422, "diagnosis 필드 형식이 올바르지 않습니다.") from e


def _save_upload_to_tempfile(file: UploadFile) -> str | None:
    """이미지 업로드 → 사전 검증(P3-1, 비이미지 400 조기 차단) 후 tempfile 경로."""
    if file is None or not file.filename:
        return None
    raw = load_validated_bytes(file)  # 크기 캡(413)·픽셀 폭탄(400)·디코드 실패(400)
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        return tmp.name


@router.post("/prescriptions", response_model=Prescription)
def create_prescription(
    question: str = Form(...),
    diagnosis: str | None = Form(None, description="이미 계산된 진단 결과(JSON 문자열) — 있으면 재사용"),
    caller_ref: str | None = Form(
        None, max_length=64,
        description="이력 테넌시 태깅용 optional 호출자 식별자(smartfarm_ai#66, 최대 64자) — 미전달 시 기존과 동일(None)",
    ),
    file: UploadFile | None = File(None),
) -> Prescription:
    diag = _parse_diagnosis(diagnosis)
    image_path: str | None = None

    try:
        with inference_slot():  # 서버 혼잡 시 429(P2-2) — 디코드·검증도 CPU를 쓰므로 슬롯 안에서
            image_path = _save_upload_to_tempfile(file)
            return prescribe_fast(question, image_path=image_path, diag=diag, caller_ref=caller_ref)
    finally:
        if image_path:
            Path(image_path).unlink(missing_ok=True)
