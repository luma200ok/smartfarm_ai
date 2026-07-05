"""app/ui.py — 라이트 모드 네이티브 텍스트 가독성 버그 수정(이슈 #48 P2).

config.toml이 base="dark"+textColor(다크용 밝은 회색 #E9F2EA)로 고정돼 있어, 라이트 토글 시에도
markdown 본문·위젯 라벨·st.metric·st.table 텍스트가 그대로 남아 흰 배경에서 안 보이던 버그를
고쳤다. inject_css()가 생성하는 CSS 문자열에 필요한 규칙·제외 패턴이 들어있는지로 회귀를
검증한다(브라우저 렌더 검증은 claude-in-chrome으로 별도 수동 확인 — 여기서는 문자열 단위)."""
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _import_ui_module():
    for p in (ROOT / "src", ROOT / "app", ROOT / "app" / "views"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("ui")


def _rendered_css(monkeypatch, ui_mod, theme):
    monkeypatch.setattr(ui_mod, "current_theme", lambda: theme)
    captured = {}
    monkeypatch.setattr(ui_mod.st, "markdown",
                         lambda html, unsafe_allow_html=False: captured.setdefault("html", html))
    ui_mod.inject_css()
    return captured["html"]


def test_inject_css_forces_markdown_container_text_to_ink(monkeypatch):
    ui_mod = _import_ui_module()
    css = _rendered_css(monkeypatch, ui_mod, "light")
    assert '[data-testid="stMarkdownContainer"]' in css
    assert "color: var(--sf-ink) !important" in css


def test_inject_css_excludes_custom_sf_components_from_ink_override(monkeypatch):
    """sf-chip/sf-kpi-label 등 커스텀 컴포넌트(이미 팔레트에서 자체 색을 받음)는
    :not([class*="sf-"])로 제외돼야 ink로 덮이지 않는다."""
    ui_mod = _import_ui_module()
    css = _rendered_css(monkeypatch, ui_mod, "light")
    assert ':not([class*="sf-"])' in css


def test_inject_css_covers_table_and_metric_testids(monkeypatch):
    """st.table(td/th)·st.metric(label/value)은 stMarkdownContainer로 안 감싸이는 별도
    testid라 각각 명시돼야 한다."""
    ui_mod = _import_ui_module()
    css = _rendered_css(monkeypatch, ui_mod, "light")
    assert '[data-testid="stTable"] td' in css
    assert '[data-testid="stTable"] th' in css
    assert '[data-testid="stMetricLabel"]' in css
    assert '[data-testid="stMetricValue"]' in css


def test_inject_css_caption_still_muted_not_forced_ink(monkeypatch):
    """캡션(stCaptionContainer)은 이 수정과 무관하게 기존 --sf-mut 규칙을 그대로 유지해야
    한다(본문과 시각적으로 구분되는 보조 텍스트) — stCaptionContainer 규칙 자체에
    --sf-ink가 아니라 --sf-mut이 쓰이는지 직접 확인."""
    ui_mod = _import_ui_module()
    css = _rendered_css(monkeypatch, ui_mod, "light")
    idx = css.index('[data-testid="stCaptionContainer"]')
    caption_rule = css[idx: idx + 80]
    assert "var(--sf-mut)" in caption_rule


def _b_specificity(selector: str) -> int:
    """class(.foo)·attribute([foo=bar]) 개수 합("b" 특이성) — :not(...) 안의 셀렉터도
    스펙상 그대로 합산되므로 괄호만 제거하고 센다. 이 프로젝트가 비교 대상 두 셀렉터에서
    id·type 셀렉터를 쓰지 않으므로 이 근사치로 충분하다."""
    flat = selector.replace(":not(", "").replace(")", "")
    return len(re.findall(r"\.[\w-]+", flat)) + len(re.findall(r"\[[^\]]+\]", flat))


def test_inject_css_tab_label_descendant_inherits_button_color_instead_of_ink_override(monkeypatch):
    """리뷰 P1 픽스(이슈 #48) — Streamlit 탭은
    <button data-baseweb="tab"><div data-testid="stMarkdownContainer"><p>라벨</p></div></button>
    구조라, 네이티브 텍스트 가독성 규칙의 '[data-testid="stMarkdownContainer"] *:not(...)'가
    <p>에 직접 매치돼 버튼의 color(선택=흰/비선택=accent) 상속을 막아버렸다(상속 vs 직접매치는
    specificity와 무관하게 항상 직접매치가 이김 — 이전 버전의 "!important 소스순서로 보호"
    주석은 캐스케이드 오해였음).

    단순 문자열 존재 확인은 이 회귀를 못 잡는다는 지적을 받아, 실제로 탭 전용 규칙
    ('.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] * { color: inherit }')의
    b-특이성이 일반 ink 오버라이드 셀렉터보다 높은지 직접 계산해 검증한다 — 그래야 탭 내부
    markdown 요소가 ink 대신 버튼의 color를 상속받을 수 있다."""
    ui_mod = _import_ui_module()
    css = _rendered_css(monkeypatch, ui_mod, "light")

    ink_override_selector = '[data-testid="stMarkdownContainer"] *:not([class*="sf-"])'
    tab_guard_selector = '.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] *'
    assert ink_override_selector in css
    assert tab_guard_selector in css

    # 탭 전용 규칙이 더 높은 specificity를 가져야 !important 동급 상황에서 이긴다.
    assert _b_specificity(tab_guard_selector) > _b_specificity(ink_override_selector)

    # 실제로 그 셀렉터의 선언이 "부모 color 상속"(하드코딩 색이 아님)인지 — 그래야 선택/비선택
    # 양쪽 탭에서 버튼이 이미 갖고 있는 올바른 색(흰색/accent)을 그대로 물려받는다.
    tab_guard_rule = css[css.index(tab_guard_selector):css.index(tab_guard_selector) + 120]
    assert "color: inherit !important" in tab_guard_rule

    # 탭 버튼 자체의 색 규칙(선택/비선택)은 그대로 유지돼야 한다(회귀 방지).
    assert "color: var(--sf-accent) !important" in css  # 비선택 탭
    assert "color: white !important" in css              # 선택 탭


def test_inject_css_tab_guard_also_covers_markdown_container_itself(monkeypatch):
    """2차 회귀(실제 브라우저 렌더로 발견) — descendant(" *")에만 inherit를 걸었더니 여전히
    ink로 보였다: <p>의 직접 부모인 stMarkdownContainer DIV 자신은 일반 오버라이드의
    '[data-testid="stMarkdownContainer"]:not(...)' (컨테이너 자체를 직접 매치하는 항목)에
    그대로 걸려 ink가 되고, <p>는 그 DIV로부터 "inherit"하니 결국 ink가 이어졌다.
    그래서 tab-guard 규칙은 컨테이너 자신도 별도 셀렉터 항목으로 포함해야 한다 — 디센던트
    셀렉터의 문자열 접두어로 우연히 매치되는 게 아니라, 콤마로 구분된 독립 항목인지까지
    확인한다(그래야 진짜 이 규칙이 컨테이너 자체에도 적용된다)."""
    ui_mod = _import_ui_module()
    css = _rendered_css(monkeypatch, ui_mod, "light")

    ink_container_selector = '[data-testid="stMarkdownContainer"]:not([class*="sf-"])'
    tab_guard_container_selector = '.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"]'
    # 컨테이너 전용 항목이 콤마로 끝나는 "독립된" 셀렉터로 존재하는지 — descendant(" *") 항목의
    # 접두어로만 우연히 매치되는 게 아님을 보장(이게 없으면 이전 회귀가 재발한다).
    assert f"{tab_guard_container_selector}," in css
    assert _b_specificity(tab_guard_container_selector) > _b_specificity(ink_container_selector)


def test_inject_css_works_for_both_themes(monkeypatch):
    """다크에서도 동일한 규칙이 주입돼 텍스트가 계속 밝게 유지되는지(대칭) 확인."""
    ui_mod = _import_ui_module()
    css_dark = _rendered_css(monkeypatch, ui_mod, "dark")
    css_light = _rendered_css(monkeypatch, ui_mod, "light")
    for css in (css_dark, css_light):
        assert "color: var(--sf-ink) !important" in css
        assert '[data-testid="stMarkdownContainer"]' in css
