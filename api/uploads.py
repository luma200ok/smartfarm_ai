"""업로드 파일 공통 검증 헬퍼 — `diagnoses`·`prescriptions` 라우터 공용(code-reviewer P1/P2).

3코어 CPU 서버 자체 DoS 방지 — 업로드 크기·픽셀 수 상한을 명시적으로 강제한다. 에러 상세는
서버 로그로만 남기고(P2-3), 클라이언트에는 고정 문구만 응답한다.
"""
import io
import logging

from fastapi import UploadFile
from PIL import Image

from .errors import ApiError

_log = logging.getLogger(__name__)

# 10MB — 잎 사진 1장(보통 수 MB)엔 여유롭고, 대용량 업로드로 인한 메모리·디스크 낭비(DoS)를 막는다.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Pillow 기본 Image.MAX_IMAGE_PIXELS(~89.5M)는 경고(DecompressionBombWarning)만 하고 통과시켜
# "압축은 작지만 디코드하면 거대한" 픽셀 폭탄 방어로는 불충분 — 이 서비스가 다루는 잎 사진
# 해상도(수백만~수천만 px)를 넉넉히 넘는 2500만 px을 명시 상한으로 둔다.
MAX_IMAGE_PIXELS = 25_000_000


def _read_capped(file: UploadFile) -> bytes:
    """`MAX_UPLOAD_BYTES` 초과분은 전부 읽기 전에 413으로 차단(전체를 메모리에 올리지 않음)."""
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "파일이 너무 큽니다(최대 10MB).")
    return raw


def _decode_capped(file: UploadFile) -> tuple[bytes, Image.Image]:
    """크기 캡 + 디코드 + 픽셀 폭탄 방어. 원본 바이트와 디코드된 PIL 이미지를 함께 돌려준다.

    Exception을 넓게 잡는다 — ultralytics가 임포트 시 `PIL.Image.open`을 몽키패치해 HEIF
    폴백을 시도하는데(pi-heif 미설치 시 `ModuleNotFoundError`), 사용자가 올린 임의 바이트의
    디코드 실패는 원인과 무관하게 모두 400이어야 한다(500 아님).
    """
    raw = _read_capped(file)
    if not raw:
        raise ApiError(400, "빈 파일입니다.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()  # 지연 디코드를 여기서 강제 — 손상 이미지도 이 시점에 예외로 잡는다
    except Exception as e:
        _log.warning("이미지 디코드 실패: %s: %s", type(e).__name__, e)
        raise ApiError(400, "유효한 이미지 파일이 아닙니다.") from e
    if pil.size[0] * pil.size[1] > MAX_IMAGE_PIXELS:
        raise ApiError(400, "이미지 해상도가 너무 큽니다(최대 2500만 픽셀).")
    return raw, pil


def load_pil(file: UploadFile) -> Image.Image:
    """`/api/diagnoses`용 — 검증된 RGB PIL 이미지."""
    _raw, pil = _decode_capped(file)
    return pil.convert("RGB")


def load_validated_bytes(file: UploadFile) -> bytes:
    """`/api/prescriptions`용(P3-1) — 디코드 가능함을 사전 검증한 원본 바이트(tempfile 저장용).

    비이미지 업로드가 tempfile 저장 → DL 진단 → LLM 호출까지 흘러가는 낭비를 여기서 차단한다.
    """
    raw, _pil = _decode_capped(file)
    return raw
