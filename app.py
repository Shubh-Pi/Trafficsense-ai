"""
app.py — TrafficSense AI entry point
Detects mobile vs desktop using a JS → query_params bridge on first load,
then routes to mobile_ui.py or desktop_ui.py exclusively.

FIX: st.markdown() silently strips <script> tags — JS never ran.
     Now uses st.components.v1.html() which actually executes JS.
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
    ("is_mobile",      None),   # None = not yet detected
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Device detection ───────────────────────────────────────────────────────────
# FIX: Use st.components.v1.html() — the ONLY way to run JS in Streamlit.
# st.markdown() strips all <script> tags regardless of unsafe_allow_html=True.

params = st.query_params

if st.session_state.is_mobile is None:
    raw_w = params.get("w", None)
    if raw_w is not None:
        try:
            w = int(raw_w)
            st.session_state.is_mobile = (w <= 768)
            st.query_params.clear()
            st.rerun()
        except ValueError:
            st.session_state.is_mobile = False
            st.rerun()
    else:
        # Show loading screen CSS via markdown (safe — no script tags)
        st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background: #080a10 !important; }
/* Hide Streamlit chrome while loading */
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
</style>
<div style="display:flex;align-items:center;justify-content:center;height:90vh;background:#080a10;">
  <div style="text-align:center;">
    <div style="font-size:48px;margin-bottom:16px;">🚦</div>
    <div style="font-family:'Inter',sans-serif;font-size:16px;font-weight:600;color:#f0f0f0;">
      TrafficSense<span style="color:#e24b4a;">AI</span>
    </div>
    <div style="font-size:12px;color:#4b5563;margin-top:8px;">Detecting device…</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── THE ACTUAL FIX: components.html() executes JS ──────────────────
        # Reads window.innerWidth and redirects with ?w= so Streamlit reruns.
        components.html("""
<!DOCTYPE html>
<html>
<body>
<script>
(function() {
    var w = window.innerWidth ||
            window.parent.innerWidth ||
            screen.width ||
            1024;                         // safe fallback

    // Walk up to the top Streamlit frame to get the real URL
    var targetWindow = window.parent || window;
    var currentUrl = targetWindow.location.href;

    var url = new URL(currentUrl);
    url.searchParams.set('w', String(w));

    // Navigate the parent frame (the actual Streamlit page)
    targetWindow.location.href = url.toString();
})();
</script>
</body>
</html>
""", height=0)          # height=0 → invisible iframe, no layout shift

        st.stop()

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
    # ── MOBILE ──────────────────────────────────────────────────────────────
    import mobile_ui as ui
    st.markdown(ui.MOBILE_CSS, unsafe_allow_html=True)
    ui.render_navbar(mk, both_ok)
    ui.render_model_strip(mk)
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
    # ── DESKTOP ──────────────────────────────────────────────────────────────
    import desktop_ui as ui
    st.markdown(ui.DESKTOP_CSS, unsafe_allow_html=True)
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