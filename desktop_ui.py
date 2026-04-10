"""
desktop_ui.py — Desktop layout for TrafficSense AI
Horizontal tab row at top. No mobile nav. No bottom bar.
"""
import io, datetime, time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

from shared import (
    DANGER_ICONS, DANGER_COLORS, NO_SIGN_THRESHOLD, MAX_HISTORY,
    get_info, preprocess_image, predict, make_csv,
    badge, conf_ind, render_warns, render_pred, render_no_sign,
    render_top3, draw_sparkline, draw_gauge, BASE_CSS
)

DESKTOP_CSS = BASE_CSS + """
<style>
/* ── Desktop-specific ─────────────────────────────────────────── */
/* Button row pulled up to sit inside the navbar */
.nav-btn-row {
  position: relative;
  z-index: 200;
  margin-top: -52px;      /* pull up into the 58px navbar */
  padding: 0 8px 0 220px; /* left-pad to clear the brand logo */
  display: flex;
  align-items: center;
  pointer-events: auto;
}
/* Make Streamlit's block container inside .nav-btn-row transparent */
.nav-btn-row > div[data-testid="stHorizontalBlock"] {
  width: 100%;
  background: transparent !important;
}
/* Remove default Streamlit top padding that would shift rows down */
.nav-btn-row .block-container { padding-top: 0 !important; }
/* Compact the buttons so they fit in 58px navbar height */
.nav-btn-row .stButton > button {
  padding: 6px 14px !important;
  font-size: 12px !important;
  height: 34px !important;
  line-height: 1 !important;
}
.nav-right { display: flex; align-items: center; gap: 10px; }
/* Ensure no leftover mobile chrome */
.mobile-only { display: none !important; }
</style>
"""

DESKTOP_CSS_LIGHT = BASE_CSS + """
<style>
/* ══════════════════════════════════════════════════
   LIGHT MODE — full override of every dark rule
   All selectors need !important to beat BASE_CSS
   ══════════════════════════════════════════════════ */

/* ── Page background & text ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
section.main { background: #f0f2f8 !important; color: #111827 !important; }

/* ── Navbar ── */
.navbar { background: #ffffff !important; border-bottom: 1px solid #d1d5db !important; }
.nav-brand { color: #111827 !important; }
.nav-brand span { color: #e24b4a !important; }
.nav-sub { color: #6b7280 !important; }
.nav-logo-icon { background: rgba(226,75,74,0.1) !important; border-color: rgba(226,75,74,0.3) !important; }

/* ── Cards ── */
.card { background: #ffffff !important; border-color: #e5e7eb !important; }
.card-body { color: #111827 !important; }
.ct  { background: linear-gradient(90deg,#e24b4a,rgba(226,75,74,0.15),transparent) !important; }
.ctb { background: linear-gradient(90deg,#3b82f6,rgba(59,130,246,0.15),transparent) !important; }
.ctg { background: linear-gradient(90deg,#22c55e,rgba(34,197,94,0.15),transparent) !important; }

/* ── Dropzone ── */
.dz { background: #ffffff !important; border-color: #d1d5db !important; }
.dz:hover { border-color: rgba(226,75,74,0.4) !important; }
.dz-title { color: #111827 !important; }
.dz-sub   { color: #6b7280 !important; }
.dz-fmt   { color: #9ca3af !important; }

/* ── Stats grid ── */
.sg { background: transparent !important; }
.sc { background: #ffffff !important; border-color: #e5e7eb !important; }
.sl { color: #6b7280 !important; }
.sv { color: #111827 !important; }

/* ── Labels & text ── */
.slabel   { color: #6b7280 !important; }
.ik       { color: #6b7280 !important; }
.iv       { color: #111827 !important; }
.ba-label { color: #6b7280 !important; }
.ptime    { background: rgba(34,197,94,0.07) !important; border-color: rgba(34,197,94,0.25) !important; color: #16a34a !important; }

/* ── Info rows ── */
.ir { border-bottom-color: #e5e7eb !important; }
.ir:last-child { border-bottom: none !important; }

/* ── Progress / confidence bars ── */
.pb-bg { background: #e5e7eb !important; }
.conf-big { border-top-color: #e5e7eb !important; border-bottom-color: #e5e7eb !important; color: #111827 !important; }

/* ── Prediction boxes ── */
.mbox { background: rgba(226,75,74,0.05) !important; border-left-color: #e24b4a !important; }
.abox { background: rgba(34,197,94,0.05) !important; border-left-color: #22c55e !important; }
.blabel { color: inherit !important; }
.btext  { color: #111827 !important; }
.pred-name { color: #111827 !important; }
.pred-sub  { color: #6b7280 !important; }

/* ── Live tab ── */
.live-name { color: #111827 !important; }
.live-conf { color: #111827 !important; }
.latrow    { color: #6b7280 !important; }

/* ── No-sign ── */
.nosign      { background: #ffffff !important; border-color: #d1d5db !important; }
.nosign-tips { background: #f9fafb !important; border-color: #e5e7eb !important; }
.ntip        { color: #6b7280 !important; }

/* ── Top-3 bars ── */
.bn { color: #374151 !important; }
.bb { background: #e5e7eb !important; }
.bp { color: #374151 !important; }

/* ── Tables ── */
.comptbl th { color: #374151 !important; border-bottom-color: #e5e7eb !important; background: #f9fafb !important; }
.comptbl td { color: #374151 !important; border-bottom-color: #e5e7eb !important; }
.met        { color: #374151 !important; }
.htbl th    { color: #374151 !important; border-bottom-color: #e5e7eb !important; background: #f9fafb !important; }
.htbl td    { color: #374151 !important; border-bottom-color: #e5e7eb !important; }
.hname      { color: #111827 !important; }

/* ── History most-detected ── */
.most-det       { background: linear-gradient(135deg,rgba(226,75,74,0.06),rgba(226,75,74,0.01)) !important; border-color: rgba(226,75,74,0.2) !important; }
.most-det-label { color: #6b7280 !important; }
.most-det-name  { color: #111827 !important; }

/* ── Per-class bars ── */
.cab-name { color: #374151 !important; }
.cab-bg   { background: #e5e7eb !important; }

/* ── Confidence indicators ── */
.ci-h { background: rgba(34,197,94,0.06) !important; border-color: rgba(34,197,94,0.25) !important; color: #16a34a !important; }
.ci-l { background: rgba(245,158,11,0.06) !important; border-color: rgba(245,158,11,0.25) !important; color: #d97706 !important; }
.wbanner { background: rgba(245,158,11,0.05) !important; border-color: rgba(245,158,11,0.25) !important; color: #92400e !important; }
.mwarn   { background: rgba(168,85,247,0.05) !important; border-color: rgba(168,85,247,0.2) !important;  color: #7c3aed !important; }

/* ── Streamlit widget overrides ── */
.stButton > button[kind="secondary"] {
  background: #ffffff !important;
  color: #374151 !important;
  border: 1px solid #d1d5db !important;
}
.stButton > button[kind="secondary"]:hover {
  color: #111827 !important;
  border-color: #9ca3af !important;
}
div[data-testid="stCameraInput"] {
  background: #ffffff !important;
  border-color: #e5e7eb !important;
}
.stCheckbox label { color: #374151 !important; }
div[data-testid="stDownloadButton"] > button {
  background: #ffffff !important;
  color: #16a34a !important;
  border-color: rgba(34,197,94,0.3) !important;
}
/* File uploader area */
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border-color: #e5e7eb !important;
}
/* Selectbox / text input */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div {
  background: #ffffff !important;
  color: #111827 !important;
  border-color: #d1d5db !important;
}
/* Model info section sidebar */
.nav-right { display: flex; align-items: center; gap: 10px; }
.mobile-only { display: none !important; }
</style>
"""

def render_navbar(mk, both_ok):
    if "is_dark" not in st.session_state:
        st.session_state.is_dark = True

    status_color = "#22c55e" if both_ok else "#f59e0b"
    status_text  = "Both Models Ready" if both_ok else "Partial Load"
    tab          = st.session_state.get("tab", 0)
    theme_icon   = "🌙" if st.session_state.is_dark else "☀️"

    def mk_cls(key): return "model-btn active" if mk == key else "model-btn"
    def tab_cls(i):  return "tab-btn active"   if tab == i  else "tab-btn"

    # ── Pure HTML navbar (visual only) ────────────────────────────────────────
    st.markdown(f"""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🚦</div>
    <div>
      <div class="nav-brand">TrafficSense<span>AI</span></div>
      <div class="nav-sub">Sign Recognition · ADAS · EfficientNet-B3</div>
    </div>
  </div>
  <div class="nav-center" id="nav-center-placeholder"></div>
  <div class="nav-right">
    <div class="status-pill">
      <div class="sdot" style="background:{status_color};"></div>{status_text}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Real st.button row — pulled up over the navbar via CSS ────────────────
    # CSS class .nav-btn-row uses margin-top:-58px + position:relative + z-index
    st.markdown('<div class="nav-btn-row">', unsafe_allow_html=True)
    c1, c2, _sep, c3, c4, c5, c6, c7 = st.columns([1.1, 1.1, 0.25, 1, 1, 1, 1, 0.35])
    with c1:
        if st.button("🇮🇳 Indian", key="nav_indian",
                     type="primary" if mk == "indian" else "secondary"):
            st.session_state.mk = "indian"; st.rerun()
    with c2:
        if st.button("🌍 GTSRB", key="nav_gtsrb",
                     type="primary" if mk == "gtsrb" else "secondary"):
            st.session_state.mk = "gtsrb"; st.rerun()
    with c3:
        if st.button("📤 Upload", key="d_upload",
                     type="primary" if tab == 0 else "secondary"):
            st.session_state.tab = 0; st.rerun()
    with c4:
        if st.button("📷 Live", key="d_live",
                     type="primary" if tab == 1 else "secondary"):
            st.session_state.tab = 1; st.rerun()
    with c5:
        if st.button("📊 Stats", key="d_stats",
                     type="primary" if tab == 2 else "secondary"):
            st.session_state.tab = 2; st.rerun()
    with c6:
        if st.button("🕒 History", key="d_history",
                     type="primary" if tab == 3 else "secondary"):
            st.session_state.tab = 3; st.rerun()
    with c7:
        if st.button(theme_icon, key="d_theme", type="secondary",
                     help="Toggle light / dark mode"):
            st.session_state.is_dark = not st.session_state.is_dark
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_tab_row(mk, tab):
    """No-op: navbar owns everything now."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
def render_upload_tab(model, mok, mk, mk_label, labels):
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    cl, cr = st.columns([1, 1], gap="large")
    with cl:
        st.markdown(f'<div class="slabel">Active Model — {mk_label}</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload", type=["jpg","jpeg","png","webp","bmp","tiff","tif"],
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if not uploaded_files:
            st.markdown("""
<div class="dz">
  <div class="dz-icon">🚦</div>
  <div class="dz-title">Drop traffic sign images here</div>
  <div class="dz-sub">or click Browse files above — up to 5 images at once</div>
  <div class="dz-fmt">PNG · JPG · JPEG · WEBP · BMP · TIFF</div>
</div>""", unsafe_allow_html=True)
        else:
            files = uploaded_files[:5]
            if len(files) == 1:
                f = files[0]
                from PIL import Image
                raw = Image.open(f)
                proc, warns, changed = preprocess_image(raw)
                if changed:
                    st.markdown('<div class="slabel">Preprocessing Preview</div>', unsafe_allow_html=True)
                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown('<div class="ba-label">BEFORE</div>', unsafe_allow_html=True)
                        st.image(raw.convert("RGB").resize((224,224)), use_column_width=True)
                    with b2:
                        st.markdown('<div class="ba-label">AFTER</div>', unsafe_allow_html=True)
                        st.image(proc, use_column_width=True)
                else:
                    st.image(proc, use_column_width=True)
                render_warns(warns)
                if mok:
                    with st.spinner("Classifying…"):
                        ids, probs, ms = predict(model, proc)
                    st.markdown(f'<div class="ptime">⚡ {ms:.0f} ms · {mk_label}</div>', unsafe_allow_html=True)
                    if probs[0] < NO_SIGN_THRESHOLD:
                        render_no_sign(probs[0])
                    else:
                        render_pred(ids[0], probs[0], mk, mk_label, labels)
                        render_top3(ids, probs, labels, mk)
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.upload_history.insert(0, {
                            "name": labels.get(str(ids[0]), f"Class {ids[0]}")[:30],
                            "conf": round(probs[0]*100, 1),
                            "danger": get_info(mk, int(ids[0]))["danger"],
                            "model": mk, "time": ts,
                            "top2": labels.get(str(ids[1]), ""),
                            "top3": labels.get(str(ids[2]), "")
                        })
                        st.session_state.upload_history = st.session_state.upload_history[:MAX_HISTORY]
                        share_text = f"TrafficSense AI: {labels.get(str(ids[0]), f'Class {ids[0]}')} ({probs[0]*100:.1f}%) | {get_info(mk, int(ids[0]))['action']}"
                        st.code(share_text, language=None)
            else:
                st.markdown(f'<div class="slabel">Batch Results — {len(files)} images</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(files), 3))
                from PIL import Image
                for i, f in enumerate(files):
                    with cols[i % 3]:
                        raw = Image.open(f); proc, warns, _ = preprocess_image(raw)
                        st.image(proc, use_column_width=True, caption=f.name[:18])
                        if warns: render_warns(warns[:1])
                        if mok:
                            with st.spinner(""):
                                ids, probs, ms = predict(model, proc)
                            if probs[0] < NO_SIGN_THRESHOLD:
                                st.markdown('<div style="font-size:12px;color:#4b5563;text-align:center;padding:6px;">🔍 No sign</div>', unsafe_allow_html=True)
                            else:
                                name = labels.get(str(ids[0]), f"Class {ids[0]}")
                                info = get_info(mk, int(ids[0]))
                                color = DANGER_COLORS.get(info["danger"], "#22c55e")
                                icon  = DANGER_ICONS.get(info["danger"], "🟢")
                                st.markdown(f'<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:8px;padding:10px;margin-top:4px;text-align:center;"><div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">{name[:20]}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:700;color:{color};">{probs[0]*100:.1f}%</div><div style="font-size:10px;margin-top:4px;">{icon} {info["danger"]}</div></div>', unsafe_allow_html=True)
                                ts = datetime.datetime.now().strftime("%H:%M:%S")
                                st.session_state.upload_history.insert(0, {
                                    "name": name[:30], "conf": round(probs[0]*100,1),
                                    "danger": info["danger"], "model": mk, "time": ts,
                                    "top2": labels.get(str(ids[1]),""),
                                    "top3": labels.get(str(ids[2]),"")
                                })
                st.session_state.upload_history = st.session_state.upload_history[:MAX_HISTORY]
    with cr:
        _render_upload_sidebar(mk, mk_label)
    st.markdown('</div>', unsafe_allow_html=True)

def _render_upload_sidebar(mk, mk_label):
    from shared import load_summary
    summary = load_summary()
    i_sum = summary.get(f"{mk}_model", summary)
    t1  = i_sum.get("final_val_acc",   76.81 if mk=="indian" else 83.42)
    t3  = i_sum.get("final_top3_acc",  92.80 if mk=="indian" else 97.09)
    n_up   = len(st.session_state.upload_history)
    avg_c  = np.mean([h["conf"] for h in st.session_state.upload_history]) if st.session_state.upload_history else 0.0
    st.markdown(f"""<div class="sg">
  <div class="sc"><div class="sl">Top-1 Val Acc</div><div class="sv" style="color:#22c55e;">{t1:.1f}%</div></div>
  <div class="sc"><div class="sl">Top-3 Val Acc</div><div class="sv" style="color:#3b82f6;">{t3:.1f}%</div></div>
  <div class="sc"><div class="sl">Images Classified</div><div class="sv">{n_up}</div></div>
  <div class="sc"><div class="sl">Avg Confidence</div><div class="sv" style="color:#e24b4a;">{avg_c:.1f}%</div></div>
</div>""", unsafe_allow_html=True)
    n_cls = 59 if mk=="indian" else 43
    ds    = "Indian Signs (~13K)" if mk=="indian" else "GTSRB European (50K+)"
    rows  = [("Architecture","EfficientNet-B3"),("Dataset",ds),("Classes",str(n_cls)),
             ("Backbone","ImageNet pretrained"),("Training","2-Phase Transfer Learning"),
             ("Preprocessing","CLAHE · Bilateral · Unsharp")]
    st.markdown('<div class="slabel" style="margin-top:4px;">Model Info</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-body">', unsafe_allow_html=True)
    for k, v in rows:
        st.markdown(f'<div class="ir"><span class="ik">{k}</span><span class="iv">{v}</span></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_live_tab(model, mok, mk, mk_label, labels):
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(226,75,74,0.05);border:1px solid rgba(226,75,74,0.15);border-radius:10px;padding:11px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;"><span style="font-size:20px;">📡</span><div><div style="font-size:13px;font-weight:600;color:#93c5fd;">Live Autodetect — {mk_label}</div><div style="font-size:11px;color:#4b5563;margin-top:2px;">Point at a traffic sign · Auto-capture runs continuously · Hold steady for 2 seconds</div></div></div>', unsafe_allow_html=True)
    cl, cr = st.columns([1.1, 1], gap="large")
    with cl:
        st.markdown('<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:10px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;"><span>🎯</span><span style="font-size:11px;color:#4b5563;">Centre the sign in frame — it should fill at least 60% of view.</span></div>', unsafe_allow_html=True)
        cam = st.camera_input("Camera", label_visibility="collapsed", key=f"cam_{mk}")
        ca1, ca2 = st.columns(2)
        with ca1: auto = st.checkbox("🔄 Auto-refresh", value=not st.session_state.freeze)
        with ca2:
            if st.button("❄️ Freeze" if not st.session_state.freeze else "▶️ Resume", key="frz", type="secondary"):
                st.session_state.freeze = not st.session_state.freeze; st.rerun()
        if cam and mok and not st.session_state.freeze:
            from PIL import Image
            raw = Image.open(cam); proc, _, _ = preprocess_image(raw)
            with st.spinner("Detecting…"):
                ids, probs, ms = predict(model, proc)
            no_sign = probs[0] < NO_SIGN_THRESHOLD
            st.session_state.live_result = {"ids":ids,"probs":probs,"ms":ms,"no_sign":no_sign,"cid":int(ids[0]),"conf":float(probs[0])}
            if not no_sign:
                st.session_state.sparkline.append(float(probs[0]))
                st.session_state.sparkline = st.session_state.sparkline[-12:]
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.webcam_history.insert(0, {
                    "name": labels.get(str(ids[0]), f"Class {ids[0]}")[:30],
                    "conf": round(probs[0]*100,1),
                    "danger": get_info(mk, int(ids[0]))["danger"],
                    "model": mk, "time": ts
                })
                st.session_state.webcam_history = st.session_state.webcam_history[:MAX_HISTORY]
        r = st.session_state.live_result
        if r:
            st.markdown(f'<div class="ptime">⚡ {r["ms"]:.0f} ms · {mk_label}{"  · ❄️ Frozen" if st.session_state.freeze else ""}</div>', unsafe_allow_html=True)
            if r["no_sign"]:
                st.markdown(f'<div style="background:#0d0f18;border:1px dashed #1a1d2e;border-radius:12px;padding:28px 20px;text-align:center;"><div style="font-size:34px;margin-bottom:10px;">🔍</div><div style="font-size:15px;font-weight:600;color:#4b5563;">No Sign Detected</div><div style="font-size:11px;color:#374151;margin-top:8px;">Confidence {r["conf"]*100:.1f}% — Centre the sign and hold steady</div></div>', unsafe_allow_html=True)
            else:
                cid = r["cid"]; conf = r["conf"]
                name = labels.get(str(cid), f"Class {cid}")
                info = get_info(mk, cid); color = DANGER_COLORS.get(info["danger"], "#22c55e")
                st.markdown(f"""<div class="card"><div class="ct"></div><div class="card-body">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div>
      <div style="font-size:9px;color:#4b5563;letter-spacing:1px;margin-bottom:4px;">DETECTED SIGN</div>
      <div class="live-name">{name}</div>
      <div style="margin-top:6px;">{badge(info['danger'])}</div>
    </div>
  </div>
  {conf_ind(conf)}
  <div class="live-conf" style="color:{color};">{conf*100:.1f}%</div>
  <div class="abox" style="margin-top:0;">
    <div class="blabel" style="color:#22c55e;">ADAS Action</div>
    <div class="btext" style="color:#86efac;">{info['action']}</div>
  </div>
  <div class="latrow"><span>{mk_label}</span><span>Latency: {r['ms']:.0f}ms</span></div>
</div></div>""", unsafe_allow_html=True)
        if auto and cam and not st.session_state.freeze:
            time.sleep(0.35); st.rerun()
    with cr:
        r = st.session_state.live_result
        if r and not r["no_sign"]: render_top3(r["ids"], r["probs"], labels, mk)
        if st.session_state.sparkline:
            st.markdown('<div class="slabel">Confidence Trend</div>', unsafe_allow_html=True)
            sp = draw_sparkline(st.session_state.sparkline)
            if sp: st.image(sp, use_column_width=True)
        wh    = st.session_state.webcam_history
        n_det = len(wh)
        avg_c = np.mean([h["conf"] for h in wh]) if wh else 0
        st.markdown(f"""<div class="card"><div class="card-body">
  <div class="slabel">Session Stats</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:38px;font-weight:700;color:#22c55e;">{n_det}</div>
  <div style="font-size:11px;color:#4b5563;margin-bottom:10px;">signs detected</div>
  <div style="height:4px;background:#1a1d2e;border-radius:2px;margin-bottom:10px;"><div style="width:{min(n_det*5,100)}%;height:4px;background:#22c55e;border-radius:2px;"></div></div>
  <div style="font-size:11px;color:#4b5563;">Avg confidence: <span style="color:#f0f0f0;font-family:'JetBrains Mono',monospace;">{avg_c:.1f}%</span></div>
</div></div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear Webcam History", key="cwh", type="secondary"):
            st.session_state.webcam_history = []
            st.session_state.live_result    = None
            st.session_state.sparkline      = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_stats_tab(mk):
    from shared import load_summary
    import io as _io
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    summary = load_summary()
    cl, cr  = st.columns([1, 1], gap="large")
    i_s = summary.get("indian_model", {}); g_s = summary.get("gtsrb_model", {})
    i_t1=i_s.get("final_val_acc",76.81);   i_t3=i_s.get("final_top3_acc",92.80)
    i_p1=i_s.get("phase1_best_val_acc",49.19); i_p2=i_s.get("phase2_best_val_acc",76.81)
    g_t1=g_s.get("final_val_acc",83.42);   g_t3=g_s.get("final_top3_acc",97.09)
    g_p1=g_s.get("phase1_best_val_acc",39.53); g_p2=g_s.get("phase2_best_val_acc",83.42)
    with cl:
        st.markdown('<div class="slabel">Model Accuracy Gauges</div>', unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        with gc1:
            g1 = draw_gauge(i_t1, "#93c5fd", "🇮🇳 Indian Top-1")
            if g1: st.image(g1, use_column_width=True)
        with gc2:
            g2 = draw_gauge(g_t1, "#fca5a5", "🌍 GTSRB Top-1")
            if g2: st.image(g2, use_column_width=True)
        st.markdown('<div class="slabel" style="margin-top:4px;">Model Comparison</div>', unsafe_allow_html=True)
        comp = [
            ("Top-1 Val Accuracy",f"{i_t1:.1f}%",f"{g_t1:.1f}%"),
            ("Top-3 Val Accuracy",f"{i_t3:.1f}%",f"{g_t3:.1f}%"),
            ("Phase 1 Best",f"{i_p1:.1f}%",f"{g_p1:.1f}%"),
            ("Phase 2 Best",f"{i_p2:.1f}%",f"{g_p2:.1f}%"),
            ("Classes","59","43"),("Training Images","~13K","50K+"),
            ("Architecture","EfficientNet-B3","EfficientNet-B3")
        ]
        rows_h = "".join(f'<tr><td class="met">{m}</td><td class="vv" style="color:#93c5fd;">{iv}</td><td class="vv" style="color:#fca5a5;">{gv}</td></tr>' for m,iv,gv in comp)
        st.markdown(f'<div class="card"><div class="card-body"><table class="comptbl"><thead><tr><th>Metric</th><th style="text-align:center;">🇮🇳 Indian</th><th style="text-align:center;">🌍 GTSRB</th></tr></thead><tbody>{rows_h}</tbody></table></div></div>', unsafe_allow_html=True)
        # Training curves
        st.markdown('<div class="slabel" style="margin-top:4px;">Training Accuracy Curves</div>', unsafe_allow_html=True)
        ih1=i_s.get("training_history",{}).get("phase1",{}); ih2=i_s.get("training_history",{}).get("phase2",{})
        gh1=g_s.get("training_history",{}).get("phase1",{}); gh2=g_s.get("training_history",{}).get("phase2",{})
        i_val=ih1.get("val_acc",[])+ih2.get("val_acc",[]); g_val=gh1.get("val_acc",[])+gh2.get("val_acc",[])
        if not i_val: i_val=[28,31,37,35,41,37,41,43,41,45,45,45,43,42,46,45,48,49,48,49,49,48,47,48,47,57,61,65,67,70,69,71,71,73,74,75,75,75,76,76,76,76,76]
        if not g_val: g_val=[19,25,30,28,34,30,34,34,33,36,34,36,37,37,37,38,38,38,38,38,38,38,39,39,39,49,54,58,61,65,68,71,73,75,77,79,80,81,81,82,81,82,83]
        p1i = len(ih1.get("val_acc", i_val[:25]))
        fig, ax = plt.subplots(figsize=(7,3.2)); fig.patch.set_facecolor("#0d0f18"); ax.set_facecolor("#080a10")
        ax.axvspan(0.5,p1i+0.5,alpha=0.04,color="#3b82f6"); ax.axvline(p1i+0.5,color="#1a1d2e",linestyle=":",lw=1)
        ax.plot(range(1,len(i_val)+1),i_val,color="#93c5fd",lw=2,label="🇮🇳 Indian",marker="o",markersize=2)
        ax.plot(range(1,len(g_val)+1),g_val,color="#fca5a5",lw=2,label="🌍 GTSRB",marker="o",markersize=2)
        ax.text(p1i/2,min(min(i_val),min(g_val))-5,"Phase 1",ha="center",fontsize=8,color="#3b82f6",style="italic")
        ax.text(p1i+(len(i_val)-p1i)/2,min(min(i_val),min(g_val))-5,"Phase 2",ha="center",fontsize=8,color="#f59e0b",style="italic")
        ax.set_xlabel("Epoch",fontsize=9,color="#4b5563"); ax.set_ylabel("Val Acc (%)",fontsize=9,color="#4b5563")
        ax.tick_params(colors="#4b5563",labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor("#1a1d2e")
        ax.grid(True,color="#1a1d2e",alpha=0.7,lw=0.5); ax.legend(fontsize=9,labelcolor="#d1d5db",facecolor="#0d0f18",edgecolor="#1a1d2e")
        plt.tight_layout(); buf=_io.BytesIO(); plt.savefig(buf,format="png",dpi=130,bbox_inches="tight",facecolor="#0d0f18"); buf.seek(0); plt.close()
        st.image(buf, use_column_width=True)
        # Training time breakdown
        st.markdown('<div class="slabel" style="margin-top:4px;">Phase Training Time Breakdown</div>', unsafe_allow_html=True)
        p1i_ep=i_s.get("phase1_epochs",25); p2i_ep=i_s.get("phase2_epochs",20)
        p1g_ep=g_s.get("phase1_epochs",25); p2g_ep=g_s.get("phase2_epochs",20)
        p1_m=[p1i_ep*108/60,p1g_ep*300/60]; p2_m=[p2i_ep*149/60,p2g_ep*416/60]
        fig2, ax2 = plt.subplots(figsize=(7,1.8)); fig2.patch.set_facecolor("#0d0f18"); ax2.set_facecolor("#080a10")
        x = np.arange(2)
        ax2.barh(x,p1_m,color="#3b82f6",label="Phase 1",height=0.4)
        ax2.barh(x,p2_m,left=p1_m,color="#e24b4a",label="Phase 2",height=0.4)
        ax2.set_yticks(x); ax2.set_yticklabels(["🇮🇳 Indian","🌍 GTSRB"],fontsize=9,color="#9ca3af")
        ax2.set_xlabel("Minutes",fontsize=9,color="#4b5563"); ax2.tick_params(colors="#4b5563",labelsize=8)
        for s in ax2.spines.values(): s.set_edgecolor("#1a1d2e")
        ax2.grid(True,axis="x",color="#1a1d2e",alpha=0.7,lw=0.5); ax2.legend(fontsize=8,labelcolor="#d1d5db",facecolor="#0d0f18",edgecolor="#1a1d2e",loc="lower right")
        plt.tight_layout(); buf2=_io.BytesIO(); plt.savefig(buf2,format="png",dpi=130,bbox_inches="tight",facecolor="#0d0f18"); buf2.seek(0); plt.close()
        st.image(buf2, use_column_width=True)
    with cr:
        for m_key, m_sum, color, label_t in [("indian",i_s,"#93c5fd","🇮🇳 Indian"),("gtsrb",g_s,"#fca5a5","🌍 GTSRB")]:
            pc = m_sum.get("per_class_accuracy", {})
            if pc:
                sc = sorted(pc.items(), key=lambda x: x[1], reverse=True)
                st.markdown(f'<div class="slabel" style="margin-top:4px;">{label_t} — Top 5 Classes</div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-body">', unsafe_allow_html=True)
                for name, acc in sc[:5]:
                    st.markdown(f'<div class="cab"><span class="cab-name">{name[:22]}</span><div class="cab-bg"><div class="cab-fill" style="width:{min(acc,100):.0f}%;background:{color};"><span class="cab-pct">{acc:.0f}%</span></div></div></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="slabel" style="margin-top:2px;">{label_t} — Bottom 5 Classes</div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-body">', unsafe_allow_html=True)
                for name, acc in sc[-5:]:
                    st.markdown(f'<div class="cab"><span class="cab-name">{name[:22]}</span><div class="cab-bg"><div class="cab-fill" style="width:{min(acc,100):.0f}%;background:#e24b4a;"><span class="cab-pct">{acc:.0f}%</span></div></div></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            else:
                t1_ = i_t1 if m_key=="indian" else g_t1; t3_ = i_t3 if m_key=="indian" else g_t3
                p1_ = i_p1 if m_key=="indian" else g_p1; p2_ = i_p2 if m_key=="indian" else g_p2
                n_  = 59 if m_key=="indian" else 43; ds_ = "Indian Signs (~13K)" if m_key=="indian" else "GTSRB (50K+)"
                st.markdown(f'<div class="slabel" style="margin-top:4px;">{label_t}</div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-body">', unsafe_allow_html=True)
                for k2, v2 in [("Architecture","EfficientNet-B3"),("Dataset",ds_),("Classes",str(n_)),
                               ("Top-1 Acc",f"{t1_:.2f}%"),("Top-3 Acc",f"{t3_:.2f}%"),
                               ("Phase 1 Best",f"{p1_:.2f}%"),("Phase 2 Best",f"{p2_:.2f}%")]:
                    vc = color if "Acc" in k2 or "Best" in k2 else "#f0f0f0"
                    st.markdown(f'<div class="ir"><span class="ik">{k2}</span><span class="iv" style="color:{vc};">{v2}</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="slabel" style="margin-top:2px;">Shared Config</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-body">', unsafe_allow_html=True)
        for k2, v2 in [("Framework","PyTorch + torchvision"),("Optimiser","AdamW"),
                       ("Scheduler","CosineAnnealingLR"),("Regularisation","Mixup + Label Smoothing"),
                       ("Preprocessing","CLAHE · Bilateral · NLM · Unsharp"),("Hardware","Kaggle P100 GPU")]:
            st.markdown(f'<div class="ir"><span class="ik">{k2}</span><span class="iv">{v2}</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_history_tab():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    uh = st.session_state.upload_history
    wh = st.session_state.webcam_history
    all_h = uh + wh
    dc = {"MANDATORY":"#fca5a5","WARNING":"#fcd34d","INFO":"#86efac"}
    if all_h:
        counts   = Counter(h["name"] for h in all_h)
        top_sign, top_count = counts.most_common(1)[0]
        top_info = next((h for h in all_h if h["name"] == top_sign), {})
        icon  = DANGER_ICONS.get(top_info.get("danger","INFO"), "🟢")
        color = DANGER_COLORS.get(top_info.get("danger","INFO"), "#22c55e")
        st.markdown(f'<div class="most-det"><div class="most-det-icon">{icon}</div><div><div class="most-det-label">Most Detected Sign This Session</div><div class="most-det-name">{top_sign}</div></div><div class="most-det-count" style="color:{color};">{top_count}×</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">📤 Upload History</div>', unsafe_allow_html=True)
    if uh:
        fc1, fc2, fc3 = st.columns([2,1,1])
        with fc1: su = st.text_input("Search", placeholder="Filter by name…", key="su", label_visibility="collapsed")
        with fc2: fdu = st.selectbox("Type", ["All","MANDATORY","WARNING","INFO"], key="fdu", label_visibility="collapsed")
        with fc3: fmu = st.selectbox("Model", ["All","indian","gtsrb"], key="fmu", label_visibility="collapsed")
        fu = [h for h in uh if (su.lower() in h["name"].lower() if su else True) and (h["danger"]==fdu if fdu!="All" else True) and (h["model"]==fmu if fmu!="All" else True)]
        st.markdown('<div class="card"><div class="card-body"><table class="htbl"><thead><tr><th>Sign</th><th>Confidence</th><th>Type</th><th>Model</th><th>Time</th></tr></thead><tbody>', unsafe_allow_html=True)
        for h in fu[:50]:
            m_ = "🇮🇳" if h["model"]=="indian" else "🌍"
            c_ = dc.get(h["danger"],"#86efac"); ico = DANGER_ICONS.get(h["danger"],"🟢")
            st.markdown(f'<tr><td class="hname">{h["name"]}</td><td class="hconf" style="color:{c_};">{h["conf"]:.1f}%</td><td>{ico} {h["danger"]}</td><td>{m_} {h["model"].upper()}</td><td style="color:#374151;font-family:\'JetBrains Mono\',monospace;font-size:10px;">{h["time"]}</td></tr>', unsafe_allow_html=True)
        st.markdown('</tbody></table></div></div>', unsafe_allow_html=True)
        if len(fu) > 50: st.markdown(f'<div style="font-size:11px;color:#374151;margin-top:-10px;margin-bottom:10px;">Showing 50 of {len(fu)} results</div>', unsafe_allow_html=True)
        csv_u = make_csv([[h["name"],h["conf"],h["danger"],h["model"],h["time"],h.get("top2",""),h.get("top3","")] for h in uh], ["Sign","Confidence%","Type","Model","Time","Top2","Top3"])
        st.download_button("⬇️ Export Upload History CSV", csv_u, "upload_history.csv", "text/csv", key="dlu")
    else:
        st.markdown('<div class="card"><div class="card-body" style="text-align:center;padding:28px;color:#374151;">No upload history yet</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="slabel">📷 Webcam Detection History</div>', unsafe_allow_html=True)
    if wh:
        fc1b, fc2b, fc3b = st.columns([2,1,1])
        with fc1b: sw = st.text_input("Search", placeholder="Filter by name…", key="sw", label_visibility="collapsed")
        with fc2b: fdw = st.selectbox("Type", ["All","MANDATORY","WARNING","INFO"], key="fdw", label_visibility="collapsed")
        with fc3b: fmw = st.selectbox("Model", ["All","indian","gtsrb"], key="fmw", label_visibility="collapsed")
        fw = [h for h in wh if (sw.lower() in h["name"].lower() if sw else True) and (h["danger"]==fdw if fdw!="All" else True) and (h["model"]==fmw if fmw!="All" else True)]
        st.markdown('<div class="card"><div class="card-body"><table class="htbl"><thead><tr><th>Sign</th><th>Confidence</th><th>Type</th><th>Model</th><th>Time</th></tr></thead><tbody>', unsafe_allow_html=True)
        for h in fw[:50]:
            m_ = "🇮🇳" if h["model"]=="indian" else "🌍"
            c_ = dc.get(h["danger"],"#86efac"); ico = DANGER_ICONS.get(h["danger"],"🟢")
            st.markdown(f'<tr><td class="hname">{h["name"]}</td><td class="hconf" style="color:{c_};">{h["conf"]:.1f}%</td><td>{ico} {h["danger"]}</td><td>{m_} {h["model"].upper()}</td><td style="color:#374151;font-family:\'JetBrains Mono\',monospace;font-size:10px;">{h["time"]}</td></tr>', unsafe_allow_html=True)
        st.markdown('</tbody></table></div></div>', unsafe_allow_html=True)
        csv_w = make_csv([[h["name"],h["conf"],h["danger"],h["model"],h["time"]] for h in wh], ["Sign","Confidence%","Type","Model","Time"])
        st.download_button("⬇️ Export Webcam History CSV", csv_w, "webcam_history.csv", "text/csv", key="dlw")
        if st.button("🗑 Clear All History", key="clrall", type="secondary"):
            st.session_state.upload_history = []
            st.session_state.webcam_history = []
            st.session_state.live_result    = None
            st.session_state.sparkline      = []
            st.rerun()
    else:
        st.markdown('<div class="card"><div class="card-body" style="text-align:center;padding:28px;color:#374151;">No webcam detections yet — only confident detections (&gt;30%) are logged</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)