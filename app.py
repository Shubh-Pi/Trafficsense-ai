"""
app.py — TrafficSense AI entry point
HF Spaces Fix: The previous JS approach failed because HF Spaces embeds
Streamlit in a cross-origin iframe — window.parent is blocked by the browser,
so the ?w= redirect never fired.

Solution: Use st.context.headers to read the User-Agent server-side (no JS,
no iframes, no redirects). Falls back gracefully → desktop layout.
"""
import streamlit as st

st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state defaults ─────────────────────────────────────────────────────
for k, v in [
    ("tab",            0),
    ("mk",             "gtsrb"),
    ("upload_history", []),
    ("webcam_history", []),
    ("live_result",    None),
    ("freeze",         False),
    ("sparkline",      []),
    ("is_mobile",      None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

_qp = st.query_params
if "mk"  in _qp: st.session_state.mk  = _qp["mk"]
if "tab" in _qp: st.session_state.tab = int(_qp["tab"])

# ── Device detection via User-Agent (server-side, no JS required) ──────────────
def detect_mobile_from_ua() -> bool:
    try:
        headers = st.context.headers
        ua = headers.get("user-agent", "") or headers.get("User-Agent", "")
        ua_lower = ua.lower()
        mobile_keywords = ("mobile", "android", "iphone", "ipad", "ipod",
                           "blackberry", "windows phone", "opera mini")
        
        # Also check screen width hint if available
        is_mobile_ua = any(kw in ua_lower for kw in mobile_keywords)
        
        # HF Spaces sometimes passes viewport hints in other headers
        sec_ch_ua_mobile = headers.get("sec-ch-ua-mobile", "?0")
        is_mobile_hint = sec_ch_ua_mobile == "?1"
        
        return is_mobile_ua or is_mobile_hint
    except Exception:
        return False

if st.session_state.is_mobile is None:
    st.session_state.is_mobile = detect_mobile_from_ua()

# Sidebar hidden by BASE_CSS. Force layout via ?layout=mobile or ?layout=desktop
_lp = st.query_params.get("layout", None)
if _lp == "mobile":   st.session_state.is_mobile = True
elif _lp == "desktop": st.session_state.is_mobile = False

# ── Load models (shared) ───────────────────────────────────────────────────────
from shared import load_models, load_all_labels

models_dict = load_models()
all_labels  = load_all_labels()

mk         = st.session_state.mk
model, mok = models_dict[mk]
labels     = all_labels[mk]
both_ok    = models_dict["indian"][1] and models_dict["gtsrb"][1]
mk_label   = "🇮🇳 Indian (59 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"
tab        = st.session_state.tab

if not mok:
    st.error(f"❌ {mk_label} weights not found. Place the .pth file alongside app.py.")

# ── Route to correct layout ────────────────────────────────────────────────────
if st.session_state.is_mobile:
    import mobile_ui as ui
    st.markdown(ui.MOBILE_CSS, unsafe_allow_html=True)
    ui.render_navbar(mk, both_ok)
    # model strip removed — switcher is now inside render_navbar
    mk         = st.session_state.mk
    model, mok = models_dict[mk]
    labels     = all_labels[mk]
    mk_label   = "🇮🇳 Indian (59 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"
    tab        = st.session_state.tab
    if tab == 0:   ui.render_upload_tab(model, mok, mk, mk_label, labels)
    elif tab == 1: ui.render_live_tab(model, mok, mk, mk_label, labels)
    elif tab == 2: ui.render_stats_tab(mk)
    elif tab == 3: ui.render_history_tab()
    ui.render_bottom_nav(tab)

else:
    import desktop_ui as ui
    if "is_dark" not in st.session_state:
        st.session_state.is_dark = True
    css = ui.DESKTOP_CSS if st.session_state.is_dark else ui.DESKTOP_CSS_LIGHT
    st.markdown(css, unsafe_allow_html=True)
    ui.render_navbar(mk, both_ok)
    ui.render_tab_row(mk, tab)
    mk         = st.session_state.mk
    model, mok = models_dict[mk]
    labels     = all_labels[mk]
    mk_label   = "🇮🇳 Indian (59 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"
    tab        = st.session_state.tab
    st.markdown("<div style='height:2px;background:#1a1d2e;'></div>", unsafe_allow_html=True)
    if tab == 0:   ui.render_upload_tab(model, mok, mk, mk_label, labels)
    elif tab == 1: ui.render_live_tab(model, mok, mk, mk_label, labels)
    elif tab == 2: ui.render_stats_tab(mk)
    elif tab == 3: ui.render_history_tab()
    st.markdown(
        '<div style="border-top:1px solid #1a1d2e;margin-top:32px;padding:14px 28px;'
        'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        '<span style="font-size:12px;color:#374151;">TrafficSense AI — Advanced Driver Assistance System</span>'
        '<span style="font-size:11px;color:#374151;font-family:\'JetBrains Mono\',monospace;">'
        'EfficientNet-B3 · PyTorch · Streamlit</span></div>',
        unsafe_allow_html=True
    )