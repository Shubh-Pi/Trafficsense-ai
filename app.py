"""
app.py — TrafficSense AI entry point
Mobile Fix: UA detection is unreliable on HF Spaces (proxy strips headers).
Solution: JS-based screen width detection sets ?layout=mobile via query param.
Falls back to desktop if JS fails.
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
if "mk"  in _qp: st.session_state.mk  = _qp["mk"]
if "tab" in _qp: st.session_state.tab = int(_qp["tab"])

# ── Step 1: If ?layout= is not set yet, inject JS to detect screen width ───────
_layout_param = st.query_params.get("layout", None)

if _layout_param is None:
    # Inject JS — this runs client-side and reloads the page with ?layout=mobile or desktop
    components.html("""
    <script>
    (function() {
        try {
            var w = window.screen.width || window.innerWidth;
            var layout = (w < 768) ? 'mobile' : 'desktop';

            // Try to set on the parent window (works if same-origin)
            try {
                var url = new URL(window.parent.location.href);
                url.searchParams.set('layout', layout);
                window.parent.location.replace(url.toString());
            } catch(e) {
                // Cross-origin blocked — set on iframe's own location instead
                // Streamlit reads query params from its own iframe URL on HF Spaces
                var url2 = new URL(window.location.href);
                url2.searchParams.set('layout', layout);
                window.location.replace(url2.toString());
            }
        } catch(err) {
            // JS fully failed — do nothing, Python fallback handles it below
        }
    })();
    </script>
    """, height=0)
    st.stop()  # Pause rendering until the redirect completes and page reloads

# ── Step 2: Read the ?layout= param set by JS (or manually in URL) ────────────
_lp = st.query_params.get("layout", "desktop")

# Allow manual override via URL for bookmarking:  ?layout=mobile or ?layout=desktop
st.session_state.is_mobile = (_lp == "mobile")

# ── Fallback: try UA detection if somehow layout param is missing ──────────────
if _layout_param is None:
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