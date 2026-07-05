"""
공통 UI 헬퍼 — 이슈 #10 Streamlit 앱 전면 정리, 이슈 #47 "야외 가독 미니멀" 리디자인
(+ 딥그린 다크 기본/라이트 토글 톤업 확장).

모든 app/views/*.py 페이지가 공유하는 헤더·경보·상태뱃지·불가안내 렌더 유틸.
디자인 원칙: 이모지는 타이틀·섹션 헤더에만 1개, 본문 내부는 제거. 사용자 대면 문구는 "~해요"체.

디자인 토큰(이슈 #47) — 기본은 딥그린 다크(관제실 톤), 사이드바 토글로 라이트(야외 가독 톤업)
전환 가능. 기존 헬퍼(page_header/section/metric_row/status_badge/unavailable/alert_box)는
시그니처 그대로 유지(타 페이지 회귀 방지) — 스타일만 새 톤에 맞춘다.

테마 전환 메커니즘: st.session_state["sf_theme"]('dark'|'light', 기본 'dark')를 단일 소스로
inject_css()가 두 팔레트(_DARK/_LIGHT) 중 하나를 CSS 변수(--sf-*)로 주입한다. Streamlit
네이티브 테마는 .streamlit/config.toml에서 딥그린 다크로 고정하고(런타임 변경 API 없음),
라이트 선택 시엔 data-testid 기반으로 앱 배경·사이드바·페이지링크 등 네이티브 컨테이너 색을
CSS로 강제 오버라이드한다 — 이 testid들(stAppViewContainer/stSidebar/stHeader 등)은 Streamlit
1.58.0 기준 확인된 값이며 향후 버전에서 이름이 바뀔 수 있다(버전 취약 지점).
"""
import html

import streamlit as st

_ALERT_BOX = {"경고": st.error, "주의": st.warning, "정보": st.info}

# KPI/칩/경보 공용 레벨 → CSS 클래스·기본 라벨(이슈 #47)
_LEVELS = {"ok", "warn", "danger"}
_DEFAULT_SEVERITY_LABEL = {"ok": "정상", "warn": "주의", "danger": "경고"}

# ── 테마 팔레트 2벌(다크 기본/라이트) — 값 출처는 코디네이터 시안(dark_dash.html/tone_up.html) ──
_DARK = {
    "bg": "#0F1C15", "sidebar_bg": "#0F1C15", "ink": "#E9F2EA", "mut": "#8FA595",
    "line": "#26382C", "accent": "#5FD08A", "accent2": "#3F9D5C", "accent_soft": "#1D3527",
    "card_top": "#152520", "card_bot": "#152520",
    "header_g1": "#13291D", "header_g2": "#0F2118", "header_border": "#26382C",
    "badge_bg": "#5FD08A", "badge_text": "#08130C",
    "kpi_value": "#FFFFFF", "shortcut_icon_text": "#08130C",
    "shadow": "0 2px 10px rgba(0, 0, 0, .35)",
    "levels": {
        "ok": {"bg1": "#16261C", "bg2": "#16261C", "border": "#26382C",
                "sev_bg": "#5FD08A", "sev_text": "#08130C", "msg": "#BFEAD0",
                "chip_bg": "#1D3527", "chip_text": "#7FE0A0", "icon_bg": "#1D3527",
                "kpi_top": "#152520", "kpi_bot": "#152520", "kpi_border_top": "#5FD08A"},
        "warn": {"bg1": "#2A2214", "bg2": "#2A2214", "border": "#4A3A1E",
                 "sev_bg": "#FFB454", "sev_text": "#1A140A", "msg": "#FFD9A0",
                 "chip_bg": "#3A2F14", "chip_text": "#FFB454", "icon_bg": "#3A2F14",
                 "kpi_top": "#231D14", "kpi_bot": "#231D14", "kpi_border_top": "#FFB454"},
        "danger": {"bg1": "#2A1616", "bg2": "#2A1616", "border": "#4A2A2A",
                   "sev_bg": "#FF6B6B", "sev_text": "#1A0F0F", "msg": "#FFB4B4",
                   "chip_bg": "#3A2020", "chip_text": "#FF9B9B", "icon_bg": "#3A2020",
                   "kpi_top": "#231717", "kpi_bot": "#231717", "kpi_border_top": "#FF6B6B"},
    },
}
_LIGHT = {
    "bg": "#FFFFFF", "sidebar_bg": "#F8FCF3", "ink": "#141A0E", "mut": "#66735A",
    "line": "#E6EBE0", "accent": "#3F7D23", "accent2": "#5BA82F", "accent_soft": "#EEF5E6",
    "card_top": "#F8FCF3", "card_bot": "#FFFFFF",
    "header_g1": "#F2F9EA", "header_g2": "#E6F4D8", "header_border": "#DCEBC9",
    "badge_bg": "#3F7D23", "badge_text": "#FFFFFF",
    "kpi_value": "#141A0E", "shortcut_icon_text": "#FFFFFF",
    "shadow": "0 1px 3px rgba(20, 40, 10, .06)",
    "levels": {
        "ok": {"bg1": "#EAF3E0", "bg2": "#FFFFFF", "border": "#DCEBC9",
                "sev_bg": "#3F7D23", "sev_text": "#FFFFFF", "msg": "#1E3A12",
                "chip_bg": "#E7F3DE", "chip_text": "#2F6A1A", "icon_bg": "#EEF5E6",
                "kpi_top": "#F8FCF3", "kpi_bot": "#FFFFFF", "kpi_border_top": "#3F7D23"},
        "warn": {"bg1": "#FEF6E6", "bg2": "#FFFFFF", "border": "#F5E0B8",
                 "sev_bg": "#C97A00", "sev_text": "#FFFFFF", "msg": "#8A5A00",
                 "chip_bg": "#FBEACC", "chip_text": "#8A5A00", "icon_bg": "#FBEACC",
                 "kpi_top": "#FDF8EF", "kpi_bot": "#FFFFFF", "kpi_border_top": "#C97A00"},
        "danger": {"bg1": "#FDECEC", "bg2": "#FFFFFF", "border": "#F4D0D0",
                   "sev_bg": "#D64545", "sev_text": "#FFFFFF", "msg": "#7A2020",
                   "chip_bg": "#FBE3E3", "chip_text": "#C0322F", "icon_bg": "#FBE3E3",
                   "kpi_top": "#FDF4F4", "kpi_bot": "#FFFFFF", "kpi_border_top": "#D64545"},
    },
}


def _norm_level(level: str | None) -> str:
    return level if level in _LEVELS else "ok"


def current_theme() -> str:
    """현재 선택된 테마('dark'|'light') — 기본 'dark'. render_theme_toggle() 미호출 시에도 안전."""
    return st.session_state.get("sf_theme", "dark")


def render_theme_toggle():
    """사이드바 상단 라이트/다크 테마 토글(이슈 #47 확장) — 기본은 다크.

    st.session_state['sf_theme']를 단일 소스로 두고 inject_css()가 이를 읽어 팔레트를 정한다.
    streamlit_app.py 진입점에서 inject_css() 호출 '전'에 실행해야 같은 rerun에 반영된다."""
    st.session_state.setdefault("sf_theme", "dark")
    with st.sidebar:
        is_light = st.toggle(
            "☀️ 라이트 모드",
            value=(st.session_state["sf_theme"] == "light"),
            key="sf_theme_toggle",
            help="끄면 기본 딥그린 다크 테마예요.",
        )
        st.session_state["sf_theme"] = "light" if is_light else "dark"


def inject_css():
    """디자인 시스템 CSS 주입 — 딥그린 다크(기본)/라이트 톤업 2팔레트, 이슈 #47.

    전역 CSS 변수(--sf-*)로 토큰을 노출하고, 아래 커스텀 컴포넌트 클래스를 정의한다:
    .sf-header(page_header 그라데이션 띠) · .sf-section-label(section) ·
    .sf-kpis/.sf-kpi(kpi_cards) · .sf-chip(status_chip) · .sf-alert(alert_strip) ·
    .sf-shortcut-icon + div[class*="st-key-sf_shortcut"](대시보드 기능 바로가기 카드).
    기존 탭 스타일(.stTabs)은 새 톤 색으로 맞춰 유지한다.
    """
    p = _DARK if current_theme() == "dark" else _LIGHT
    lv = p["levels"]

    level_css = []
    for name, d in lv.items():
        level_css.append(f"""
        .sf-alert.{name} {{
            background: linear-gradient(90deg, {d['bg1']}, {d['bg2']} 70%);
            border-color: {d['border']};
            border-left-color: {d['sev_bg']};
        }}
        .sf-alert.{name} .sf-alert-sev {{ background: {d['sev_bg']}; color: {d['sev_text']}; }}
        .sf-alert.{name} .sf-alert-msg {{ color: {d['msg']}; }}
        .sf-chip.{name} {{ background: {d['chip_bg']}; color: {d['chip_text']}; }}
        .sf-kpi.{name} {{
            background: linear-gradient({d['kpi_top']}, {d['kpi_bot']});
            border-top-color: {d['kpi_border_top']};
        }}
        .sf-kpi.{name} .sf-kpi-icon {{ background: {d['icon_bg']}; }}
        """)

    st.markdown(f"""
    <style>
    :root {{
        --sf-bg: {p['bg']};
        --sf-ink: {p['ink']};
        --sf-mut: {p['mut']};
        --sf-line: {p['line']};
        --sf-accent: {p['accent']};
        --sf-accent2: {p['accent2']};
        --sf-accent-soft: {p['accent_soft']};
        --sf-shadow: {p['shadow']};
    }}

    /* 전역 타이포·배경 */
    .stApp {{ background: var(--sf-bg); color: var(--sf-ink); }}
    .stApp, .stApp p, .stApp li, .stApp label {{ font-size: 15.5px; }}
    .stApp h1, .stApp h2, .stApp h3 {{ color: var(--sf-ink); letter-spacing: -.3px; }}

    /* ── 네이티브 컨테이너 오버라이드(버전 취약 — streamlit==1.58.0 기준 testid) ──
       config.toml은 딥그린 다크를 기본값으로 고정하므로(런타임 스위치 API 없음), 라이트 선택 시
       이 규칙들이 앱 배경·사이드바·헤더바까지 밝게 강제한다. */
    [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"],
    [data-testid="stBottomBlockContainer"] {{ background: var(--sf-bg) !important; }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: {p['sidebar_bg']} !important;
    }}
    [data-testid="stSidebar"] * {{ color: var(--sf-ink) !important; }}
    [data-testid="stPageLink"] * {{ color: var(--sf-ink) !important; }}
    [data-testid="stCaptionContainer"] {{ color: var(--sf-mut) !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--sf-line) !important; }}
    .stApp hr {{ border-color: var(--sf-line) !important; }}

    /* 기존 탭 스타일 — 새 톤으로 유지 */
    .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 1.15rem;
        font-weight: 700;
        padding: 12px 24px;
        background-color: var(--sf-accent-soft);
        color: var(--sf-accent);
        border-radius: 10px 10px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--sf-accent);
        color: white !important;
    }}

    /* ── 페이지 헤더(page_header) — 그린 그라데이션 띠 + 하단 accent 밑줄 + 뱃지 ── */
    .sf-header {{
        background: linear-gradient(120deg, {p['header_g1']}, {p['header_g2']});
        border: 1px solid {p['header_border']};
        border-radius: 16px;
        padding: 18px 24px 16px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }}
    .sf-header::after {{
        content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
        background: linear-gradient(90deg, var(--sf-accent), var(--sf-accent2));
    }}
    .sf-header-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .sf-header-title {{ margin: 0; font-size: 27px; font-weight: 900; letter-spacing: -.6px;
        color: var(--sf-ink); }}
    .sf-header-badge {{
        font-size: 11px; font-weight: 800; letter-spacing: .3px;
        padding: 3px 10px; border-radius: 999px;
        background: {p['badge_bg']}; color: {p['badge_text']};
    }}
    .sf-header-cap {{ margin: 6px 0 0; font-size: 12.5px; color: var(--sf-mut); }}

    /* ── 섹션 라벨(section) — accent 색 + 좌측 3px 색바 ── */
    .sf-section-label {{
        display: flex; align-items: center; gap: 8px;
        font-size: 12px; font-weight: 800; letter-spacing: .4px; text-transform: uppercase;
        color: var(--sf-accent); margin: 18px 0 9px;
    }}
    .sf-section-label::before {{
        content: ""; width: 3px; height: 14px; background: var(--sf-accent); border-radius: 2px;
    }}
    .sf-section-cap {{ margin: -4px 0 8px; font-size: 12.5px; color: var(--sf-mut); }}

    /* ── KPI 카드 그리드(kpi_cards) — 아이콘 원 + 큰 숫자 ── */
    .sf-kpis {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 6px 0 4px;
    }}
    .sf-kpi {{
        background: linear-gradient({p['card_top']}, {p['card_bot']});
        border: 1px solid var(--sf-line);
        border-top: 4px solid var(--sf-accent);
        border-radius: 13px;
        padding: 14px 16px 15px;
        box-shadow: var(--sf-shadow);
    }}
    .sf-kpi-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
    .sf-kpi-icon {{
        width: 30px; height: 30px; border-radius: 9px;
        background: var(--sf-accent-soft);
        display: flex; align-items: center; justify-content: center; font-size: 16px;
        flex: 0 0 auto;
    }}
    .sf-kpi-label {{ font-size: 12.5px; font-weight: 700; color: var(--sf-mut); }}
    .sf-kpi-value {{
        font-size: 34px;
        font-weight: 900;
        letter-spacing: -1px;
        color: {p['kpi_value']};
        font-variant-numeric: tabular-nums;
    }}
    .sf-kpi-value .sf-kpi-unit {{
        font-size: 16px;
        font-weight: 700;
        color: var(--sf-mut);
        margin-left: 3px;
    }}

    /* ── 상태 칩(status_chip) ── */
    .sf-chip {{
        display: inline-block;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: .3px;
        padding: 3px 9px;
        border-radius: 6px;
        margin-top: 8px;
    }}

    /* ── 경보 스트립(alert_strip) — 채운 그라데이션 배너 ── */
    .sf-alert {{
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid var(--sf-line);
        border-left: 5px solid var(--sf-accent);
        border-radius: 12px;
        padding: 13px 16px;
        margin-bottom: 8px;
    }}
    .sf-alert-sev {{
        flex: 0 0 auto;
        font-size: 12px;
        font-weight: 800;
        padding: 4px 11px;
        border-radius: 6px;
        letter-spacing: .4px;
    }}
    .sf-alert-msg {{ font-size: 14.5px; font-weight: 700; }}

    /* ── 기능 바로가기 카드(st.container(border=True, key="sf_shortcut_*")) ── */
    .sf-shortcut-icon {{
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, var(--sf-accent), var(--sf-accent2));
        color: {p['shortcut_icon_text']};
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; margin-bottom: 9px;
    }}
    div[class*="st-key-sf_shortcut"] {{
        border: 1px solid var(--sf-line) !important;
        border-radius: 12px !important;
        box-shadow: var(--sf-shadow);
        transition: border-color .12s, box-shadow .12s, transform .12s;
    }}
    div[class*="st-key-sf_shortcut"]:hover {{
        border-color: var(--sf-accent) !important;
        box-shadow: 0 6px 18px rgba(63, 125, 35, .16);
        transform: translateY(-2px);
    }}

    {"".join(level_css)}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, caption: str | None = None, badge: str | None = None):
    """페이지 최상단 타이틀 + 설명 캡션(각 view의 render() 첫 줄에서 호출).

    (이슈 #47) 그린 그라데이션 띠 + 하단 accent 밑줄로 렌더 — badge를 주면 타이틀 옆에
    뱃지 칩을 붙인다(예: 대시보드의 "데모" 표시). badge 생략 시 기존 페이지와 동일하게 동작."""
    badge_html = f'<span class="sf-header-badge">{html.escape(badge)}</span>' if badge else ""
    cap_html = f'<p class="sf-header-cap">{html.escape(caption)}</p>' if caption else ""
    st.markdown(
        f'<div class="sf-header">'
        f'<div class="sf-header-row"><h1 class="sf-header-title">{html.escape(title)}</h1>{badge_html}</div>'
        f'{cap_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None):
    """섹션 소제목 + 부연 캡션 — accent 색 + 좌측 3px 색바(이슈 #47 톤업)."""
    cap_html = f'<p class="sf-section-cap">{html.escape(caption)}</p>' if caption else ""
    st.markdown(f'<div class="sf-section-label">{html.escape(title)}</div>{cap_html}',
                unsafe_allow_html=True)


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


# ── 이슈 #47 — 야외 가독 미니멀 + 톤업 신규 컴포넌트 ────────────────────────

def _chip_html(text: str, level: str = "ok") -> str:
    """status_chip의 HTML 조각만 생성(kpi_cards 내부 재사용 목적, 렌더는 하지 않음)."""
    return f'<span class="sf-chip {_norm_level(level)}">{html.escape(text)}</span>'


def status_chip(text: str, level: str = "ok"):
    """단독 상태 칩 렌더 — level: 'ok'|'warn'|'danger'."""
    st.markdown(_chip_html(text, level), unsafe_allow_html=True)


def kpi_cards(items: list[dict]):
    """핵심 지표 카드 그리드 — 여러 KPI를 HTML grid로 한 번에 렌더(반응형).

    items: [{"label": str, "value": str, "unit": str|None, "icon": str|None,
             "chip": str|None, "chip_level": "ok"|"warn"|"danger"}, ...]
    아이콘 원(soft 배경) + 카드 상단 4px accent 선(레벨별 색) + 큰 숫자(34px·900·tabular-nums).
    """
    cards = []
    for item in items:
        label = html.escape(str(item.get("label", "")))
        value = html.escape(str(item.get("value", "")))
        unit = item.get("unit")
        icon = item.get("icon")
        chip = item.get("chip")
        level = _norm_level(item.get("chip_level"))

        card_cls = "sf-kpi" if level == "ok" else f"sf-kpi {level}"
        icon_html = f'<span class="sf-kpi-icon">{html.escape(str(icon))}</span>' if icon else ""
        unit_html = f'<span class="sf-kpi-unit">{html.escape(str(unit))}</span>' if unit else ""
        chip_html = _chip_html(chip, level) if chip else ""

        cards.append(
            f'<div class="{card_cls}">'
            f'<div class="sf-kpi-top">{icon_html}<span class="sf-kpi-label">{label}</span></div>'
            f'<div class="sf-kpi-value">{value}{unit_html}</div>'
            f'{chip_html}'
            f'</div>'
        )
    st.markdown(f'<div class="sf-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def alert_strip(level: str, text: str, severity_label: str | None = None):
    """경보 스트립 — 채운 그라데이션 배너(좌측 상태색 바 + 라벨 칩 + 메시지).
    level: 'ok'|'warn'|'danger'."""
    lvl = _norm_level(level)
    label = severity_label or _DEFAULT_SEVERITY_LABEL[lvl]
    st.markdown(
        f'<div class="sf-alert {lvl}">'
        f'<span class="sf-alert-sev">{html.escape(label)}</span>'
        f'<span class="sf-alert-msg">{html.escape(text)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
