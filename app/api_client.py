"""Streamlit ↔ FastAPI 서빙 API(`api/`) 공용 클라이언트 (이슈 #59 PR 3).

`SMARTFARM_API_URL` 이 설정돼 있으면 뷰(`app/views/diagnosis.py`·`app/views/prescribe.py`)가
in-process 추론 대신 이 클라이언트로 HTTP API를 호출한다. 미설정이면 뷰는 이 모듈을 아예
호출하지 않고 기존 in-process 경로를 그대로 쓴다(이 파일은 API 모드 전용).

컨벤션(다른 llm 모듈과 동일 — src/llm/weather.py 참고):
- API_BASE는 호출 시점 os.getenv()로 읽는다(모듈 상수 캐싱 금지) — 테스트 monkeypatch 대응.
- 요청 실패(연결·타임아웃·5xx)는 예외를 삼키지 않고 `ApiClientError`로 감싸 전파한다(뷰가
  st.error로 명시적으로 알리기 위함 — 조용한 in-process 폴백은 하지 않음, 핸드오프 명시).
- status 코드로 분기(메시지 매칭 금지) — 429=혼잡, 400/413=업로드 문제, 그 외 실패=연결 실패.
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# override=True: 쉘 전역 env보다 프로젝트 .env가 우선 (weather.py·prescribe.py와 동일 이유)
load_dotenv(ROOT / ".env", override=True)

_DIAG_TIMEOUT = 30
_PRESC_TIMEOUT = 120
_HEALTH_TIMEOUT = 10


class ApiClientError(Exception):
    """서빙 API 호출 실패 — 뷰가 그대로 st.error 메시지로 보여줄 수 있는 문구를 담는다."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def api_base() -> str | None:
    """`SMARTFARM_API_URL` 읽기(끝 슬래시 제거). 미설정·빈 문자열이면 None(=in-process 모드)."""
    url = os.getenv("SMARTFARM_API_URL")
    return url.rstrip("/") if url else None


def _raise_for_status(resp: requests.Response) -> None:
    if resp.status_code == 200:
        return
    if resp.status_code == 429:
        raise ApiClientError("서버가 혼잡해요. 잠시 후 다시 시도해 주세요.", status_code=429)
    if resp.status_code in (400, 413):
        detail = None
        try:
            body = resp.json()
            detail = body.get("detail") if isinstance(body, dict) else None
        except ValueError:
            pass
        msg = detail if isinstance(detail, str) else "업로드한 파일을 확인해 주세요(형식·크기)."
        raise ApiClientError(msg, status_code=resp.status_code)
    if resp.status_code == 422:
        raise ApiClientError("요청 형식이 올바르지 않아요.", status_code=422)
    raise ApiClientError("서빙 API 연결 실패", status_code=resp.status_code)


def _require_base() -> str:
    base = api_base()
    if not base:
        raise ApiClientError("SMARTFARM_API_URL이 설정되지 않았어요.")
    return base


def diagnose_remote(image_bytes: bytes, conf: float = 0.25, filename: str = "image.jpg") -> dict:
    """POST /api/diagnoses — multipart 업로드 → `DiagnosisResponse` dict 그대로 반환."""
    base = _require_base()
    try:
        resp = requests.post(
            f"{base}/api/diagnoses",
            params={"conf": conf},
            files={"file": (filename, image_bytes, "application/octet-stream")},
            timeout=_DIAG_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ApiClientError("서빙 API 연결 실패") from e
    _raise_for_status(resp)
    return resp.json()


def prescribe_remote(question: str, diagnosis: dict | None = None,
                      image_bytes: bytes | None = None, filename: str = "image.jpg") -> dict:
    """POST /api/prescriptions — multipart(question·diagnosis JSON 문자열·선택 file) → `Prescription` dict."""
    base = _require_base()
    data = {"question": question}
    if diagnosis is not None:
        data["diagnosis"] = json.dumps(diagnosis, ensure_ascii=False)
    files = {"file": (filename, image_bytes, "application/octet-stream")} if image_bytes else None
    try:
        resp = requests.post(f"{base}/api/prescriptions", data=data, files=files, timeout=_PRESC_TIMEOUT)
    except requests.RequestException as e:
        raise ApiClientError("서빙 API 연결 실패") from e
    _raise_for_status(resp)
    return resp.json()


def health_remote() -> dict:
    """GET /api/health — `HealthResponse` dict(ollama.online 등)."""
    base = _require_base()
    try:
        resp = requests.get(f"{base}/api/health", timeout=_HEALTH_TIMEOUT)
    except requests.RequestException as e:
        raise ApiClientError("서빙 API 연결 실패") from e
    _raise_for_status(resp)
    return resp.json()


def decode_png_base64(b64: str) -> np.ndarray:
    """cam_png_base64/annotated_png_base64 → RGB uint8 ndarray(st.image에 바로 전달 가능)."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
