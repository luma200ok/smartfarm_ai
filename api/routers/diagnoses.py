"""POST /api/diagnoses — 잎 사진 → OOD 게이트 → 부위 게이트 → CNN 진단(+Grad-CAM) → YOLO 검출.

파이프라인 순서는 `app/views/diagnosis.py`(thin wrapper) 현행 흐름을 그대로 따른다. 게이트가
차단해도 200 + `ood_blocked: true`(현행 뷰·`src/llm/tools.get_diagnosis` 동작과 동일 의미).

이미지 전달: `dl.infer`의 진단 함수들은 PIL.Image를 직접 받으므로(경로 기반 아님) 여기서는
tempfile 없이 업로드 바이트를 바로 디코드한다(prescriptions 라우터는 경로 기반 `get_diagnosis`를
쓰므로 tempfile 필요 — 핸드오프의 tempfile 방침은 그쪽에 해당).
"""
import base64
import io

import numpy as np
from dl import infer
from fastapi import APIRouter, File, Query, UploadFile
from PIL import Image

from ..errors import ApiError
from ..schemas import DetectionBox, DiagnosisResponse, YoloResult

router = APIRouter(tags=["diagnoses"])


def _load_pil(file: UploadFile) -> Image.Image:
    """업로드 바이트 → PIL RGB 이미지. 빈 파일·비이미지는 400(ApiError).

    Exception을 넓게 잡는다 — ultralytics가 임포트 시 `PIL.Image.open`을 몽키패치해
    HEIF 폴백을 시도하는데(pi-heif 미설치 시 `ModuleNotFoundError`), 사용자가 올린
    임의 바이트 디코드 실패는 원인과 무관하게 모두 400이어야 한다(500 아님).
    """
    raw = file.file.read()
    if not raw:
        raise ApiError(400, "빈 파일입니다.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()  # 지연 디코드를 여기서 강제 — 손상 이미지도 이 시점에 예외로 잡는다
    except Exception as e:
        raise ApiError(400, f"유효한 이미지 파일이 아닙니다: {type(e).__name__}: {e}") from e
    return pil.convert("RGB")


def _overlay_png_base64(img: np.ndarray, cam: np.ndarray) -> str:
    """원본(224,224,3 float[0,1]) 위에 jet 히트맵 반투명 합성 → PNG base64.

    `app/views/diagnosis.py:overlay()`와 동일 로직(사람이 보는 Grad-CAM 근거 이미지).
    """
    import matplotlib.cm as cm

    heat = cm.jet(cam)[..., :3]
    blended = (0.55 * img + 0.45 * heat).clip(0, 1)
    return _ndarray_to_png_base64((blended * 255).astype("uint8"))


def _ndarray_to_png_base64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@router.post("/diagnoses", response_model=DiagnosisResponse)
def create_diagnosis(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO 검출 신뢰도 임계값"),
) -> DiagnosisResponse:
    pil = _load_pil(file)

    # ① OOD 게이트 — 식물/잎이 아니면 차단
    score = infer.ood_plant_score(pil)
    if score < infer.PLANT_THRESHOLD:
        return DiagnosisResponse(ood_blocked=True, reason="식물·잎으로 보이지 않는 사진(OOD)")

    # ② 부위 게이트 — 잎이 아닌 부위(과실/꽃/줄기)면 차단
    part, _part_prob = infer.part_of(pil)
    if part != "leaf":
        return DiagnosisResponse(
            ood_blocked=True,
            reason=f"잎이 아닌 부위로 판정({infer.PART_KR.get(part, part)})",
            part=part,
        )

    # ③ 통과 → 잎 진단(Grad-CAM) + YOLO 검출
    label, prob, probs, cam, img = infer.predict_with_cam(pil)
    annotated, dets = infer.detect_annotated(pil, conf=conf)

    return DiagnosisResponse(
        ood_blocked=False,
        label=label,
        label_kr=infer.LABEL_KR[label],
        prob=float(prob),
        probs={c: float(p) for c, p in zip(infer.CLASSES, probs)},
        part=part,
        cam_png_base64=_overlay_png_base64(img, cam),
        yolo=YoloResult(
            annotated_png_base64=_ndarray_to_png_base64(annotated.astype("uint8")),
            boxes=[DetectionBox(label=lab, conf=float(c)) for lab, c in dets],
        ),
    )
