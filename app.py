"""
app.py — TrafficSense AI entry point
Mobile Fix: Combines UA detection (instant) + JS screen width (sets param on next load).
No st.stop(), no blank screens, no redirect loops.
"""
import streamlit as st
import streamlit.components.v1 as components

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
if "mk"    in _qp: st.session_state.mk       = _qp["mk"]
if "tab"   in _qp: st.session_state.tab      = int(_qp["tab"])
if "theme" in _qp:
    st.session_state.is_dark = (_qp["theme"] == "dark")
    st.query_params.pop("theme")  # consume it so it doesn't linger

# ── Step 1: Check ?layout= param first (most reliable, set by JS on prev load) ─
_lp = st.query_params.get("layout", None)

if _lp == "mobile":
    st.session_state.is_mobile = True
elif _lp == "desktop":
    st.session_state.is_mobile = False
else:
    # ?layout= not set yet — use UA detection as immediate fallback
    try:
        headers = st.context.headers
        ua = (headers.get("user-agent", "") or headers.get("User-Agent", "")).lower()
        mobile_kw = ("mobile", "android", "iphone", "ipad", "ipod",
                     "blackberry", "windows phone", "opera mini")
        is_mobile_ua = any(kw in ua for kw in mobile_kw)
        sec_ch = headers.get("sec-ch-ua-mobile", "?0")
        st.session_state.is_mobile = is_mobile_ua or (sec_ch == "?1")
    except Exception:
        st.session_state.is_mobile = False

# ── Step 2: Inject JS to set ?layout= for NEXT load (non-blocking) ────────────
# Renders the page NOW; JS quietly fixes the param on next refresh
if _lp is None:
    components.html("""
    <script>
    (function() {
        try {
            var w = window.screen.width || window.innerWidth || 1024;
            var layout = (w < 768) ? 'mobile' : 'desktop';
            var target = null;
            try { target = window.parent.location.href; } catch(e) {}
            if (!target) { try { target = window.location.href; } catch(e) {} }
            if (target) {
                var url = new URL(target);
                if (url.searchParams.get('layout') !== layout) {
                    url.searchParams.set('layout', layout);
                    try { window.parent.location.replace(url.toString()); }
                    catch(e) { window.location.replace(url.toString()); }
                }
            }
        } catch(err) {}
    })();
    </script>
    """, height=0)

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