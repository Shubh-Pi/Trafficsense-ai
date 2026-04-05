"""
app.py — TrafficSense AI entry point
Detects mobile vs desktop using a JS → query_params bridge on first load,
then routes to mobile_ui.py or desktop_ui.py exclusively.
No CSS display:none tricks — each layout renders ONLY its own buttons.
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
    ("is_mobile",      None),   # None = not yet detected
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Device detection ───────────────────────────────────────────────────────────
# Strategy: inject a tiny JS snippet that reads window.innerWidth and writes
# it to ?w= query param, then Streamlit rereuns and we check the param.
# This only runs once (when is_mobile is None).

params = st.query_params

if st.session_state.is_mobile is None:
    raw_w = params.get("w", None)
    if raw_w is not None:
        try:
            w = int(raw_w)
            st.session_state.is_mobile = (w <= 768)
            # Clean up the query param so it doesn't persist
            st.query_params.clear()
            st.rerun()
        except ValueError:
            st.session_state.is_mobile = False
    else:
        # Inject JS to write width to query param and trigger rerun
        st.markdown("""
<script>
(function detectDevice() {
  var w = window.innerWidth;
  var url = new URL(window.location.href);
  url.searchParams.set('w', w);
  // Replace history entry so back button still works
  window.history.replaceState({}, '', url.toString());
  // Trigger Streamlit rerun by dispatching a storage event
  // (Streamlit polls for URL changes via its own mechanism)
  window.location.href = url.toString();
})();
</script>
""", unsafe_allow_html=True)
        # Show a minimal loading state while JS runs
        st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#080a10!important;}
</style>
<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#080a10;">
  <div style="text-align:center;">
    <div style="font-size:38px;margin-bottom:14px;">🚦</div>
    <div style="font-family:'Inter',sans-serif;font-size:14px;font-weight:600;color:#f0f0f0;">TrafficSense<span style="color:#e24b4a;">AI</span></div>
    <div style="font-size:11px;color:#4b5563;margin-top:6px;">Loading…</div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.stop()

# ── Load models (shared) ───────────────────────────────────────────────────────
from shared import load_models, load_all_labels

models_dict  = load_models()
all_labels   = load_all_labels()

mk        = st.session_state.mk
model, mok = models_dict[mk]
labels     = all_labels[mk]
both_ok    = models_dict["indian"][1] and models_dict["gtsrb"][1]
mk_label   = "🇮🇳 Indian (59 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"
tab        = st.session_state.tab

if not mok:
    st.error(f"❌ {mk_label} weights not found. Place the .pth file alongside app.py.")

# ── Route to correct layout ────────────────────────────────────────────────────
if st.session_state.is_mobile:
    # ── MOBILE ────────────────────────────────────────────────────────────────
    import mobile_ui as ui
    st.markdown(ui.MOBILE_CSS, unsafe_allow_html=True)
    ui.render_navbar(mk, both_ok)
    ui.render_model_strip(mk)
    # Re-read mk in case model strip just changed it
    mk        = st.session_state.mk
    model, mok = models_dict[mk]
    labels     = all_labels[mk]
    mk_label   = "🇮🇳 Indian (59 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"
    tab        = st.session_state.tab
    # Tab content — single column, compact
    if tab == 0:   ui.render_upload_tab(model, mok, mk, mk_label, labels)
    elif tab == 1: ui.render_live_tab(model, mok, mk, mk_label, labels)
    elif tab == 2: ui.render_stats_tab(mk)
    elif tab == 3: ui.render_history_tab()
    # Bottom nav rendered LAST so JS can fix it to viewport bottom
    ui.render_bottom_nav(tab)

else:
    # ── DESKTOP ────────────────────────────────────────────────────────────────
    import desktop_ui as ui
    st.markdown(ui.DESKTOP_CSS, unsafe_allow_html=True)
    ui.render_navbar(mk, both_ok)
    ui.render_tab_row(mk, tab)
    # Re-read state after tab row (buttons may have changed mk or tab)
    mk        = st.session_state.mk
    model, mok = models_dict[mk]
    labels     = all_labels[mk]
    mk_label   = "🇮🇳 Indian (59 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"
    tab        = st.session_state.tab
    st.markdown("<div style='height:2px;background:#1a1d2e;'></div>", unsafe_allow_html=True)
    # Tab content
    if tab == 0:   ui.render_upload_tab(model, mok, mk, mk_label, labels)
    elif tab == 1: ui.render_live_tab(model, mok, mk, mk_label, labels)
    elif tab == 2: ui.render_stats_tab(mk)
    elif tab == 3: ui.render_history_tab()
    # Footer
    st.markdown('<div style="border-top:1px solid #1a1d2e;margin-top:32px;padding:14px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;"><span style="font-size:12px;color:#374151;">TrafficSense AI — Advanced Driver Assistance System</span><span style="font-size:11px;color:#374151;font-family:\'JetBrains Mono\',monospace;">EfficientNet-B3 · PyTorch · Streamlit</span></div>', unsafe_allow_html=True)
