"""
공통 UI 헬퍼 — 이슈 #10 Streamlit 앱 전면 정리, 이슈 #47 "야외 가독 미니멀" 리디자인.

모든 app/views/*.py 페이지가 공유하는 헤더·경보·상태뱃지·불가안내 렌더 유틸.
디자인 원칙: 이모지는 타이틀·섹션 헤더에만 1개, 본문 내부는 제거. 사용자 대면 문구는 "~해요"체.

디자인 토큰(이슈 #47) — 순백 바탕·고대비 잉크 + 큰 숫자·선명한 상태색으로 야외(직사광선)에서도
읽히게 함. 기존 헬퍼(page_header/section/metric_row/status_badge/unavailable/alert_box)는
시그니처 그대로 유지(타 페이지 회귀 방지) — 스타일만 새 톤에 맞춘다.
"""
import html

import streamlit as st

_ALERT_BOX = {"경고": st.error, "주의": st.warning, "정보": st.info}

# KPI/칩/경보 공용 레벨 → CSS 클래스·기본 라벨(이슈 #47)
_LEVELS = {"ok", "warn", "danger"}
_DEFAULT_SEVERITY_LABEL = {"ok": "정상", "warn": "주의", "danger": "경고"}


def _norm_level(level: str | None) -> str:
    return level if level in _LEVELS else "ok"


def inject_css():
    """디자인 시스템 CSS 주입 — 야외 가독 미니멀(순백·고대비 + 큰 숫자·선명한 상태색, 이슈 #47).

    전역 CSS 변수(--sf-*)로 토큰을 노출하고, 아래 커스텀 컴포넌트 클래스를 정의한다:
    .sf-kpis/.sf-kpi(kpi_cards) · .sf-chip(status_chip) · .sf-alert(alert_strip) ·
    .sf-shortcut(대시보드 기능 바로가기 st.page_link 카드 래핑, key="sf_shortcut_*").
    기존 탭 스타일(.stTabs)은 새 토큰 색으로 맞춰 유지한다.
    """
    st.markdown("""
    <style>
    :root {
        --sf-bg: #FFFFFF;
        --sf-ink: #141A0E;
        --sf-mut: #6D7A62;
        --sf-line: #E6EBE0;
        --sf-accent: #3F7D23;
        --sf-accent-soft: #EEF5E6;
        --sf-ok: #3F7D23;
        --sf-warn: #C97A00;
        --sf-danger: #D64545;
        --sf-shadow: 0 1px 3px rgba(20, 40, 10, .06);
    }

    /* 전역 타이포·배경 — 순백·고대비 */
    .stApp { background: var(--sf-bg); color: var(--sf-ink); }
    .stApp, .stApp p, .stApp li, .stApp label { font-size: 15.5px; }
    .stApp h1, .stApp h2, .stApp h3 { color: var(--sf-ink); letter-spacing: -.3px; }
    .stApp [data-testid="stCaptionContainer"] { color: var(--sf-mut); }

    /* 기존 탭 스타일 — 새 톤으로 유지 */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 700;
        padding: 12px 24px;
        background-color: var(--sf-accent-soft);
        color: var(--sf-accent);
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--sf-accent);
        color: white !important;
    }

    /* ── KPI 카드 그리드(kpi_cards) ── */
    .sf-kpis {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 6px 0 4px;
    }
    .sf-kpi {
        background: var(--sf-bg);
        border: 1px solid var(--sf-line);
        border-top: 3px solid var(--sf-accent);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: var(--sf-shadow);
    }
    .sf-kpi.warn { border-top-color: var(--sf-warn); }
    .sf-kpi.danger { border-top-color: var(--sf-danger); }
    .sf-kpi-label {
        font-size: 12.5px;
        font-weight: 700;
        color: var(--sf-mut);
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .sf-kpi-value {
        font-size: 34px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 6px;
        color: var(--sf-ink);
        font-variant-numeric: tabular-nums;
    }
    .sf-kpi-value .sf-kpi-unit {
        font-size: 16px;
        font-weight: 700;
        color: var(--sf-mut);
        margin-left: 3px;
    }

    /* ── 상태 칩(status_chip) ── */
    .sf-chip {
        display: inline-block;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: .3px;
        padding: 3px 9px;
        border-radius: 6px;
        margin-top: 8px;
    }
    .sf-chip.ok { background: #E7F3DE; color: var(--sf-ok); }
    .sf-chip.warn { background: #FBEACC; color: #8A5A00; }
    .sf-chip.danger { background: #FBDADA; color: #8A1F1F; }

    /* ── 경보 스트립(alert_strip) ── */
    .sf-alert {
        display: flex;
        align-items: center;
        gap: 12px;
        background: var(--sf-bg);
        border: 1px solid var(--sf-line);
        border-left: 5px solid var(--sf-danger);
        border-radius: 12px;
        padding: 13px 16px;
        box-shadow: var(--sf-shadow);
        margin-bottom: 8px;
    }
    .sf-alert.warn { border-left-color: var(--sf-warn); }
    .sf-alert.ok { border-left-color: var(--sf-ok); }
    .sf-alert-sev {
        flex: 0 0 auto;
        font-size: 12px;
        font-weight: 800;
        color: #fff;
        background: var(--sf-danger);
        padding: 4px 11px;
        border-radius: 6px;
        letter-spacing: .4px;
    }
    .sf-alert.warn .sf-alert-sev { background: var(--sf-warn); }
    .sf-alert.ok .sf-alert-sev { background: var(--sf-ok); }
    .sf-alert-msg { font-size: 14.5px; font-weight: 600; color: var(--sf-ink); }

    /* ── 기능 바로가기 카드(st.container(border=True, key="sf_shortcut_*")) ── */
    div[class*="st-key-sf_shortcut"] {
        border: 1px solid var(--sf-line) !important;
        border-top: 3px solid var(--sf-accent) !important;
        border-radius: 12px !important;
        box-shadow: var(--sf-shadow);
        transition: border-color .12s, box-shadow .12s;
    }
    div[class*="st-key-sf_shortcut"]:hover {
        box-shadow: 0 6px 18px rgba(63, 125, 35, .12);
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, caption: str | None = None):
    """페이지 최상단 타이틀 + 설명 캡션(각 view의 render() 첫 줄에서 호출)."""
    st.title(title)
    if caption:
        st.caption(caption)


def section(title: str, caption: str | None = None):
    """섹션 소제목 + 부연 캡션."""
    st.subheader(title)
    if caption:
        st.caption(caption)


def metric_row(items: list[tuple[str, str, str | None]]):
    """내부 vs 외기 대비 등 metric 카드를 한 줄로. items=[(label, value, delta), ...]."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        if delta:
            col.metric(label, value, delta)
        else:
            col.metric(label, value)


def status_badge(ok: bool, label: str):
    """단일 상태 뱃지 — 정상(success)/오프라인(warning) 표시."""
    (st.success if ok else st.warning)(label)


def unavailable(feature: str, reason: str, hint: str | None = None):
    """모델·데이터 없음 등 선택 기능 비활성 안내 — 단일 포맷으로 통일."""
    msg = f"ℹ️ {feature} 사용 불가 — {reason}"
    if hint:
        msg += f" ({hint})"
    st.caption(msg)


def alert_box(level: str, text: str):
    """경고/주의/정보 3단계 알림 — 구 phase3의 중복 dict 매핑 흡수."""
    box = _ALERT_BOX.get(level, st.info)
    box(text)


# ── 이슈 #47 — 야외 가독 미니멀 신규 컴포넌트 ──────────────────────────────

def _chip_html(text: str, level: str = "ok") -> str:
    """status_chip의 HTML 조각만 생성(kpi_cards 내부 재사용 목적, 렌더는 하지 않음)."""
    return f'<span class="sf-chip {_norm_level(level)}">{html.escape(text)}</span>'


def status_chip(text: str, level: str = "ok"):
    """단독 상태 칩 렌더 — level: 'ok'|'warn'|'danger'."""
    st.markdown(_chip_html(text, level), unsafe_allow_html=True)


def kpi_cards(items: list[dict]):
    """핵심 지표 카드 그리드 — 여러 KPI를 HTML grid로 한 번에 렌더(반응형).

    items: [{"label": str, "value": str, "unit": str|None,
             "chip": str|None, "chip_level": "ok"|"warn"|"danger"}, ...]
    카드 상단 3px accent 선(레벨별 색), 숫자는 크게(34px·900·tabular-nums).
    """
    cards = []
    for item in items:
        label = html.escape(str(item.get("label", "")))
        value = html.escape(str(item.get("value", "")))
        unit = item.get("unit")
        chip = item.get("chip")
        level = _norm_level(item.get("chip_level"))

        card_cls = "sf-kpi" if level == "ok" else f"sf-kpi {level}"
        unit_html = f'<span class="sf-kpi-unit">{html.escape(str(unit))}</span>' if unit else ""
        chip_html = _chip_html(chip, level) if chip else ""

        cards.append(
            f'<div class="{card_cls}">'
            f'<div class="sf-kpi-label">{label}</div>'
            f'<div class="sf-kpi-value">{value}{unit_html}</div>'
            f'{chip_html}'
            f'</div>'
        )
    st.markdown(f'<div class="sf-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def alert_strip(level: str, text: str, severity_label: str | None = None):
    """경보 스트립 — 좌측 상태색 바 + 라벨 칩 + 메시지. level: 'ok'|'warn'|'danger'."""
    lvl = _norm_level(level)
    label = severity_label or _DEFAULT_SEVERITY_LABEL[lvl]
    st.markdown(
        f'<div class="sf-alert {lvl}">'
        f'<span class="sf-alert-sev">{html.escape(label)}</span>'
        f'<span class="sf-alert-msg">{html.escape(text)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
