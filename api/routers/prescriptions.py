"""POST /api/prescriptions — 질문(+선택 이미지 또는 기존 진단 재사용) → LLM 자연어 처방.

`llm.prescribe.prescribe_fast(question, image_path, diag)` 인터페이스를 그대로 따른다(fast-path
1-call, 서버 웜 ~16s). 이미지·진단 모두 없어도 동작(일반 질문 → DL 진단 없이 처방).

요청은 multipart/form-data로 받는다 — `file`(선택, 이미지) 하나의 엔드포인트에서 다루려면
FastAPI에서 JSON body와 업로드 파일을 함께 받을 수 없어(순수 JSON이면 파일 필드 자체가 불가),
`diagnosis`는 JSON 문자열을 담는 Form 필드로 표현한다(핸드오프의 "JSON {question, diagnosis?}"
의도는 유지하되 전송 형식만 multipart로 통일 — 이미지 없는 클라이언트도 동일 엔드포인트로 호출
가능해 분기 부담이 없다).

이미지 전달: UploadFile → tempfile 저장 후 경로 전달(`llm.tools.get_diagnosis(image_path)`가
경로 기반이라 `prescribe_fast`도 경로를 받음) — 응답 후 임시파일 정리(finally).
"""
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from llm.prescribe import Prescription, prescribe_fast

from ..errors import ApiError

router = APIRouter(tags=["prescriptions"])


@router.post("/prescriptions", response_model=Prescription)
def create_prescription(
    question: str = Form(...),
    diagnosis: str | None = Form(None, description="이미 계산된 진단 결과(JSON 문자열) — 있으면 재사용"),
    file: UploadFile | None = File(None),
) -> Prescription:
    diag = None
    if diagnosis:
        try:
            diag = json.loads(diagnosis)
        except json.JSONDecodeError as e:
            raise ApiError(422, f"diagnosis 필드가 유효한 JSON이 아닙니다: {e}") from e

    image_path: str | None = None
    if file is not None and file.filename:
        raw = file.file.read()
        if not raw:
            raise ApiError(400, "빈 파일입니다.")
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            image_path = tmp.name

    try:
        return prescribe_fast(question, image_path=image_path, diag=diag)
    finally:
        if image_path:
            Path(image_path).unlink(missing_ok=True)
