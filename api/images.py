"""ndarray → PNG base64 인코딩 공용 헬퍼 — `diagnoses`·`detections` 라우터 공용.

두 라우터 모두 `infer.detect_annotated()`가 돌려주는 RGB ndarray를 응답에 base64 PNG로
실어 보낸다(이슈 #59 PR 3-1 — 검출 전용 `/api/detections` 분리 시 diagnoses.py의 private
헬퍼를 라우터 간 직접 import하지 않도록 공용 모듈로 뺐다).
"""
import base64
import io

import numpy as np
from PIL import Image


def ndarray_to_png_base64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
