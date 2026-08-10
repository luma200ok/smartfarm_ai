"""
AI 처방 페이지 — 잎 사진 → DL 진단(라벨·확률, 근거) + LLM 처방(초보자 눈높이 자연어).

구 app/phase3_llm.py 상단부(1) 잎 사진 ~ 디스코드 전송)를 분리 이관(이슈 #10 C2).
분업: 진단=DL(resnet18·게이트·YOLO) / 설명·처방=LLM(Ollama, 모델은 .env OLLAMA_MODEL).
Ollama 미가동이어도 페이지 전체를 죽이지 않음 — 상단에 상태 뱃지만 띄우고
DL 진단 단계는 계속 동작(진단은 이 페이지에서 직접 처방을 만들 때만 필요하므로,
LLM 오프라인이면 처방 버튼 자체를 비활성화 안내).
"""
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from api_client import ApiClientError, api_base, diagnose_remote, health_remote, prescribe_remote
from state import K_LAST_DIAG, K_LAST_PRESC
from ui import page_header, section, status_badge, unavailable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env", override=True)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")   # 표시용 — 실제 사용 모델과 같은 소스(.env)
SAMPLES = ROOT / "app" / "samples"
SAMPLE_KR = {"late_blight": "🦠 잎마름역병", "leaf_mold": "🦠 잎곰팡이병",
             "normal": "🌿 정상", "tylcv": "🦠 황화잎말이바이러스"}


def _resolve_image(uploaded, sample_key):
    """업로드 파일 or 샘플 → 로컬 파일 경로(추론 tool 은 경로를 받는다). in-process 모드 전용."""
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        return tmp.name
    if sample_key:
        p = SAMPLES / f"{sample_key}.jpg"
        return str(p) if p.exists() else None
    return None


def _resolve_image_bytes(uploaded, sample_key):
    """업로드 파일 or 샘플 → (bytes, filename). API 모드 전용(`_resolve_image`의 bytes 버전 —
    tempfile 없이 그대로 multipart 전송)."""
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if sample_key:
        p = SAMPLES / f"{sample_key}.jpg"
        return (p.read_bytes(), p.name) if p.exists() else (None, None)
    return None, None


def _diag_from_api(resp: dict) -> dict:
    """`/api/diagnoses` 응답(`DiagnosisResponse`) → `tools.get_diagnosis()`와 같은 모양의 dict.

    이 페이지는 Grad-CAM·YOLO를 렌더하지 않으므로(텍스트 진단 패널만) `cam_png_base64`·`yolo`
    같은 큰 base64 페이로드는 세션 상태·`/api/prescriptions` 재전송(diagnosis 폼 필드) 양쪽에서
    뺀다 — `DiagnosisIn`이 검증하는 필드만 남긴다.
    """
    diag: dict = {"ood_blocked": bool(resp.get("ood_blocked", False))}
    for k in ("reason", "label", "label_kr", "prob", "probs", "part"):
        if resp.get(k) is not None:
            diag[k] = resp[k]
    return diag


def _ollama_status() -> tuple[bool, str | None]:
    """(online, api_error) — API 모드면 `/api/health`의 ollama 상태, 아니면 로컬 `ollama.list()`.

    api_error는 API 모드에서 서빙 자체에 연결할 수 없을 때만 채워진다(온라인/오프라인과 구분 —
    "Ollama 데몬 꺼짐"과 "서빙 API 다운"은 안내 문구가 달라야 한다).
    """
    if api_base():
        try:
            resp = health_remote()
        except ApiClientError as e:
            return False, str(e)
        return bool(resp["ollama"]["online"]), None
    import ollama
    try:
        ollama.list()
        return True, None
    except Exception:
        return False, None


def render():
    page_header("💊 AI 처방", f"DL 진단(라벨·확률)을 받아 LLM(Ollama · {MODEL})이 초보자 눈높이 처방을 만들어요. "
                              "진단은 DL, 설명·처방은 LLM이 맡아요(LLM은 진단하지 않아요).")

    online, api_error = _ollama_status()
    if api_error:
        st.error(api_error)
        unavailable("DL 진단·자연어 처방", "서빙 API에 연결할 수 없어요", "SMARTFARM_API_URL 상태를 확인하세요")
    elif not online:
        status_badge(False, f"LLM 오프라인 — Ollama 데몬이 꺼져 있어요. `ollama serve` 후 `ollama pull {MODEL}` 을 실행하세요.")
        unavailable("자연어 처방(LLM)", "Ollama 데몬 미가동", "DL 진단만은 계속 확인할 수 있어요")

    section("1) 잎 사진")
    c1, c2 = st.columns(2)
    with c1:
        uploaded = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"])
    with c2:
        choice = st.radio("또는 샘플 선택", ["(선택 안 함)", *SAMPLE_KR.keys()],
                          format_func=lambda k: SAMPLE_KR.get(k, k), horizontal=False)
    sample_key = choice if choice in SAMPLE_KR else None

    # 미리보기 (경로 만들지 않고 위젯 값으로 바로)
    if uploaded is not None:
        st.image(uploaded, caption="입력 잎 사진", width=300)
    elif sample_key:
        st.image(str(SAMPLES / f"{sample_key}.jpg"), caption=f"샘플 · {SAMPLE_KR[sample_key]}", width=300)

    question = st.text_input("2) 물어볼 말",
                             value="이 토마토 잎 좀 봐줘. 병이면 어떻게 조치해야 해?")

    has_image = uploaded is not None or sample_key is not None
    btn_label = "💊 처방 받기" if online else "🔬 DL 진단만 받기"
    # 이슈 #15 C3 — 단계별 진행 표시(st.status). 이슈 #10 톤(에러 3단계, "~해요"체) 준수.
    _STAGE_LABEL = {
        "diagnosis": "🔬 잎 진단 실행 중…",
        "rag": "📚 근거 검색 중…",
        "forecast": "🌦️ 예보 확인 중…",
        "writing": "✍️ 처방 작성 중…",
    }
    if st.button(btn_label, type="primary", disabled=not has_image):
        if api_base():
            # API 모드(SMARTFARM_API_URL 설정) — /api/diagnoses → /api/prescriptions, 폴백 안 함
            image_bytes, filename = _resolve_image_bytes(uploaded, sample_key)
            if not image_bytes:
                st.error("이미지를 찾을 수 없어요.")
            else:
                with st.status("🔬 잎 진단 실행 중…", expanded=True) as status:
                    try:
                        # include_visuals=False — 이 페이지는 Grad-CAM·YOLO를 렌더하지 않으므로
                        # 서버가 그 계산을 생략하게 한다(P2-1). in-process도 tools.get_diagnosis()가
                        # infer.diagnose()(순전파만)를 쓰므로 의미는 동일하다.
                        diag_resp = diagnose_remote(image_bytes, filename=filename, include_visuals=False)
                    except ApiClientError as e:
                        status.update(label="⚠️ 진단 실패", state="error")
                        st.error(str(e))
                    else:
                        diag = _diag_from_api(diag_resp)
                        st.session_state[K_LAST_DIAG] = diag
                        if online:
                            status.update(label=_STAGE_LABEL["writing"])
                            try:
                                presc_resp = prescribe_remote(question, diagnosis=diag,
                                                               image_bytes=image_bytes, filename=filename)
                            except ApiClientError as e:
                                status.update(label="⚠️ 처방 실패", state="error")
                                st.error(str(e))
                            else:
                                from llm.prescribe import Prescription

                                st.session_state[K_LAST_PRESC] = Prescription.model_validate(presc_resp)
                                status.update(label="✅ 처방 완료", state="complete")
                        else:
                            st.session_state.pop(K_LAST_PRESC, None)
                            status.update(label="✅ DL 진단 완료", state="complete")
        else:
            image_path = _resolve_image(uploaded, sample_key)
            if not image_path:
                st.error("이미지를 찾을 수 없어요.")
            else:
                from llm import tools

                with st.status("🔬 잎 진단 실행 중…", expanded=True) as status:
                    st.session_state[K_LAST_DIAG] = tools.get_diagnosis(image_path)
                    if online:
                        # 이슈 #18 — 처방 버튼 경로는 흐름이 고정이므로 fast-path(1-call)로 전환.
                        from llm.prescribe import prescribe_fast

                        def _on_progress(stage: str) -> None:
                            status.update(label=_STAGE_LABEL.get(stage, "처리 중…"))

                        status.update(label=_STAGE_LABEL["diagnosis"])
                        st.session_state[K_LAST_PRESC] = prescribe_fast(
                            question, image_path=image_path, on_progress=_on_progress,
                            diag=st.session_state[K_LAST_DIAG])  # 위에서 이미 실행한 진단 재사용(중복 추론 방지)
                        status.update(label="✅ 처방 완료", state="complete")
                    else:
                        st.session_state.pop(K_LAST_PRESC, None)
                        status.update(label="✅ DL 진단 완료", state="complete")

    # 처방 결과 렌더(세션 보관 — rerun/전송 버튼에도 유지)
    if K_LAST_DIAG in st.session_state:
        from dl import infer
        from llm import notify
        diag = st.session_state[K_LAST_DIAG]

        section("🔬 DL 진단 (근거)")
        if diag.get("ood_blocked"):
            st.warning(f"진단 차단 — {diag.get('reason')}. 병명을 단정하지 않고 재촬영을 안내해요.")
        else:
            st.markdown(f"**진단:** {diag['label_kr']} · **신뢰도:** {diag['prob'] * 100:.0f}%")
            st.progress(min(max(diag["prob"], 0.0), 1.0))
            st.caption("클래스별 확률 · " + "  ".join(
                f"{infer.LABEL_KR[k]} {v * 100:.0f}%" for k, v in diag["probs"].items()))

    if K_LAST_PRESC in st.session_state:
        from llm import notify
        presc = st.session_state[K_LAST_PRESC]

        section("💬 LLM 처방")
        st.success(f"**{presc.진단요약}**")
        st.markdown(
            f"- **원인** — {presc.원인}\n"
            f"- **즉시 조치** — {presc.즉시조치}\n"
            f"- **예방** — {presc.예방}\n"
            f"- **재촬영 시점** — {presc.재촬영시점}"
        )
        if presc.근거출처:
            st.markdown("**📖 근거 출처 (농사로/NCPMS RAG)**")
            for s in presc.근거출처:
                st.markdown(f"- {s}")
        else:
            st.caption("ℹ️ 이 진단에 대한 재배가이드 근거를 찾지 못했어요.")

        if st.button("📣 디스코드로 보내기", key="send_presc"):
            ok, msg = notify.notify_prescription(presc)
            (st.success if ok else st.warning)(msg)


if __name__ == "__main__":
    st.set_page_config(page_title="AI 처방", page_icon="💊", layout="wide")
    render()
