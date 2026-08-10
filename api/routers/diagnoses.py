"""POST /api/diagnoses — 잎 사진 → OOD 게이트 → 부위 게이트 → CNN 진단(+Grad-CAM) → YOLO 검출.

파이프라인 순서는 `app/views/diagnosis.py`(thin wrapper) 현행 흐름을 그대로 따른다. 게이트가
차단해도 200 + `ood_blocked: true`(현행 뷰·`src/llm/tools.get_diagnosis` 동작과 동일 의미).

이미지 검증(크기 캡·픽셀 폭탄 방어·디코드)은 `api/uploads.py` 공용 헬퍼를 쓴다 — `dl.infer`의
진단 함수들은 PIL.Image를 직접 받으므로(경로 기반 아님) tempfile 없이 검증된 PIL을 바로 쓴다
(prescriptions 라우터는 경로 기반 `get_diagnosis`를 쓰므로 tempfile이 필요).

무거운 추론(Grad-CAM·YOLO)은 `api/concurrency.py`의 동시성 캡(최대 2) 안에서만 실행한다.
"""
import base64
import io

import numpy as np
from dl import infer
from fastapi import APIRouter, File, Query, UploadFile
from PIL import Image

from ..concurrency import inference_slot
from ..schemas import DetectionBox, DiagnosisResponse, YoloResult
from ..uploads import load_pil

router = APIRouter(tags=["diagnoses"])


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
    pil = load_pil(file)  # 크기 캡(413)·픽셀 폭탄(400)·디코드 실패(400) 모두 여기서 처리

    with inference_slot():  # 서버 혼잡 시 429(P2-2) — 이 블록 안만 추론 슬롯을 점유
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
