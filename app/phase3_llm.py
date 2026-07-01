"""
Phase 3 (LLM) — 자연어 처방 페이지 (3-1: Ollama function calling)

잎 사진 → DL 진단(라벨·확률, 근거) + LLM 처방(초보자 눈높이 자연어).
분업: 진단=DL(resnet18·게이트·YOLO) / 설명·처방=LLM(Ollama qwen2.5:14b).
멀티페이지: app/streamlit_app.py 가 render() 를 호출(set_page_config 는 엔트리에서 1회).

실행:  streamlit run app/streamlit_app.py   (프로젝트 루트에서)
전제:  Ollama 데몬 구동 + `ollama pull qwen2.5:14b`
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SAMPLES = ROOT / "app" / "samples"
SAMPLE_KR = {"leaf_mold": "🦠 잎곰팡이병", "normal": "🌿 정상", "tylcv": "🦠 황화잎말이바이러스"}


def _resolve_image(uploaded, sample_key):
    """업로드 파일 or 샘플 → 로컬 파일 경로(추론 tool 은 경로를 받는다)."""
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


def render():
    st.title("💬 Phase 3 · LLM — 자연어 처방")
    st.caption("DL 진단(라벨·확률)을 받아 로컬 LLM(Ollama · qwen2.5:14b)이 초보자 눈높이 처방으로. "
               "**진단=DL, 설명·처방=LLM** (LLM은 진단하지 않음).")

    # Ollama 구동 확인
    import ollama
    try:
        ollama.list()
    except Exception:
        st.error("⚠️ Ollama 데몬이 실행 중이 아니에요. Ollama 앱을 켜거나 터미널에서 `ollama serve` 후 "
                 "`ollama pull qwen2.5:14b` 를 실행해 주세요.")
        return

    st.markdown("#### 1) 잎 사진")
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
    if st.button("💊 처방 받기", type="primary", disabled=not has_image):
        image_path = _resolve_image(uploaded, sample_key)
        if not image_path:
            st.error("이미지를 찾을 수 없어요.")
            return

        from dl import infer
        from llm import tools
        from llm.prescribe import prescribe

        with st.spinner("DL 진단 + LLM 처방 생성 중… (로컬 14B, 20~40초 걸릴 수 있어요)"):
            diag = tools.get_diagnosis(image_path)
            presc = prescribe(question, image_path=image_path)

        # ── DL 진단 패널 (근거) ──
        st.markdown("#### 🔬 DL 진단 (근거)")
        if diag.get("ood_blocked"):
            st.warning(f"진단 차단 — {diag.get('reason')}. 병명을 단정하지 않고 재촬영을 안내합니다.")
        else:
            st.markdown(f"**진단:** {diag['label_kr']} · **신뢰도:** {diag['prob'] * 100:.0f}%")
            st.progress(min(max(diag["prob"], 0.0), 1.0))
            st.caption("클래스별 확률 · " + "  ".join(
                f"{infer.LABEL_KR[k]} {v * 100:.0f}%" for k, v in diag["probs"].items()))

        # ── LLM 처방 카드 ──
        st.markdown("#### 💬 LLM 처방")
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

    # ── A·B: 환경 예측 기반 일일 코치 · 조기 경보 (3-3) ──
    st.divider()
    st.subheader("🌡️ 환경 예측 기반 (LSTM)")

    @st.cache_data(ttl=600)
    def _forecast_summary():
        from llm.tools import get_forecast
        return get_forecast()

    fc = _forecast_summary()
    if fc and not fc.get("unavailable"):
        st.markdown(f"**다음날 내부온도** {fc['next_temp']}℃ ({fc['trend']}) · "
                    f"**습도위험** {fc['humidity_risk']} (최근 평균 {fc['humidity_mean']}%)")
    else:
        st.caption("환경 예측 비활성 — `python src/dl/train_lstm.py`로 LSTM 학습 필요")

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("🌅 오늘의 코치", use_container_width=True):
            from llm import pipeline
            with st.spinner("코칭 생성 중…"):
                coach = pipeline.daily_coach()
            st.success(coach.요약)
            for todo in coach.오늘_할일:
                st.markdown(f"- {todo}")
            st.caption(f"근거: {coach.근거}")
    with cc2:
        if st.button("⚠️ 조기 경보", use_container_width=True):
            from llm import pipeline
            with st.spinner("경보 판단 중…"):
                w = pipeline.early_warning()
            box = {"경고": st.error, "주의": st.warning}.get(w.경보수준, st.info)
            box(f"경보수준: {w.경보수준}" + (f" · 위험병해: {w.위험병해}" if w.위험병해 and w.위험병해 != "없음" else ""))
            st.markdown(f"- **이유** — {w.이유}")
            if w.권장조치:
                st.markdown(f"- **권장조치** — {w.권장조치}")

    st.divider()
    st.caption("환각 방어 3종: ① 신뢰도 톤 분기 · ② 게이트 차단 안내 · ③ 클래스 한정성(잎 병해 3종). "
               "진행 상황 → `docs/roadmap.md` Phase 3.")


if __name__ == "__main__":
    st.set_page_config(page_title="Phase 3 · LLM", page_icon="💬", layout="wide")
    render()
