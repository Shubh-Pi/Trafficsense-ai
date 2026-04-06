"""
mobile_ui.py — Mobile layout for TrafficSense AI
- NO top tab row
- Fixed bottom nav bar (4 buttons: Upload, Live, Stats, History)
- Model switcher in a compact top strip
- Single-column content for all tabs
"""
import io as _io, datetime, time
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

# ── Mobile-specific CSS ────────────────────────────────────────────────────────
MOBILE_CSS = BASE_CSS + """
<style>
/* ── Mobile layout ─────────────────────────────────────────────── */

/* Compact navbar for mobile */
.navbar {
  height: auto !important;
  flex-wrap: wrap !important;
  padding: 8px 14px !important;
  gap: 6px !important;
}
.nav-sub { display: none; }
.status-pill { font-size: 10px !important; padding: 4px 10px !important; }

/* Model switcher strip — top, full width, below navbar */
.model-strip {
  display: flex;
  gap: 0;
  background: #0d0f18;
  border-bottom: 1px solid #1a1d2e;
}
.model-strip .stButton > button {
  border-radius: 0 !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  color: #4b5563 !important;
  font-size: 13px !important;
  padding: 10px 0 !important;
}
.model-strip .stButton > button[kind="primary"] {
  color: #e24b4a !important;
  border-bottom: 2px solid #e24b4a !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* Content area — extra bottom padding so nav bar doesn't cover content */
.wrap {
  padding: 14px 12px 90px !important;
}

/* Page title */
.mob-page-title {
  font-size: 11px;
  font-weight: 700;
  color: #374151;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 10px 0 6px;
  margin-bottom: 8px;
  border-bottom: 1px solid #1a1d2e;
}

/* Prediction card — tighter on mobile */
.pred-name { font-size: 17px !important; }
.conf-big  { font-size: 36px !important; }
.live-conf { font-size: 44px !important; }
.live-name { font-size: 17px !important; }
.sv        { font-size: 22px !important; }
.bn        { width: 110px !important; }

/* ── Fixed bottom nav ──────────────────────────────────────────── */
/* 
  IMPORTANT: We cannot inject Streamlit buttons inside an arbitrary HTML div.
  Instead we render a real st.columns(4) row of st.button() widgets, 
  then use CSS to fix that specific row to the bottom of the viewport.
  
  We tag the container with a data attribute via JS injected in the page,
  and use a robust CSS selector to grab it.
  
  Fallback: we use the nth-of-type of the last horizontal block rendered.
  To make this reliable we render the bottom nav LAST on the page,
  wrapped in a unique sentinel div.
*/
.bottom-nav-sentinel + div[data-testid="stVerticalBlock"],
.bottom-nav-sentinel ~ div[data-testid="stVerticalBlock"] {
  /* not used — see JS approach below */
}

/* The sentinel + columns trick: 
   We render a zero-height sentinel div, then the 4-column nav row.
   JS finds the sentinel, then fixes the next sibling columns block. */
.bnav-row {
  position: fixed !important;
  bottom: 0 !important;
  left: 0 !important;
  right: 0 !important;
  background: #0d0f18 !important;
  border-top: 2px solid #1a1d2e !important;
  z-index: 9999 !important;
  padding: 4px 0 env(safe-area-inset-bottom, 8px) !important;
  display: flex !important;
}
.bnav-row [data-testid="column"] {
  display: flex !important;
  justify-content: center !important;
}
.bnav-row .stButton > button {
  background: transparent !important;
  border: none !important;
  color: #4b5563 !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  padding: 8px 4px !important;
  width: 100% !important;
  display: flex !important;
  flex-direction: row !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 6px !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  line-height: 1.3 !important;
}
.bnav-row .stButton > button:hover { opacity: 1 !important; }

/* Active tab nav button */
.bnav-active .stButton > button {
  color: #e24b4a !important;
}
</style>

<script>
// Fix the bottom nav row to the viewport bottom
// We find the div containing our sentinel class and fix its next sibling
(function fixBottomNav() {
  function doFix() {
    const sentinel = document.querySelector('.bnav-sentinel');
    if (!sentinel) { setTimeout(doFix, 200); return; }
    // Walk up to the stVerticalBlock, then find the next columns block
    let el = sentinel;
    while (el && el.getAttribute('data-testid') !== 'stVerticalBlock') {
      el = el.parentElement;
    }
    if (!el) { setTimeout(doFix, 200); return; }
    // The next sibling stHorizontalBlock is our nav row
    let sib = el.nextElementSibling;
    while (sib) {
      if (sib.getAttribute('data-testid') === 'stHorizontalBlock') {
        sib.classList.add('bnav-row');
        return;
      }
      sib = sib.nextElementSibling;
    }
    setTimeout(doFix, 200);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', doFix);
  } else {
    doFix();
  }
})();
</script>
"""


def render_navbar(mk, both_ok):
    """Compact mobile navbar — logo + buttons."""
    st.markdown(f"""
<div class="navbar" style="border-bottom:none; padding-bottom:0;">
  <div class="nav-logo">
    <div class="nav-logo-icon">🚦</div>
    <div>
      <div class="nav-brand">TrafficSense<span>AI</span></div>
    </div>
  </div>
  <div class="nav-right">
    <div class="status-pill">
      <div class="sdot"></div>{'Ready' if both_ok else 'Partial'}
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown('<div class="model-strip">', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns([1, 1, 0.4])
    with mc1:
        if st.button("🇮🇳 Indian", key="m_indian",
                     type="primary" if mk == "indian" else "secondary"):
            st.session_state.mk = "indian"; st.rerun()
    with mc2:
        if st.button("🌍 GTSRB", key="m_gtsrb",
                     type="primary" if mk == "gtsrb" else "secondary"):
            st.session_state.mk = "gtsrb"; st.rerun()
    with mc3:
        if st.button("◐", key="m_contrast", type="secondary"):
            st.session_state.is_dark = not st.session_state.get("is_dark", True)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_model_strip(mk):
    pass


def render_bottom_nav(tab):
    """
    Fixed bottom nav with 4 real Streamlit buttons.
    Rendered LAST on the page. JS + CSS fixes it to viewport bottom.
    """
    # Sentinel — JS finds this to locate the nav row
    st.markdown('<div class="bnav-sentinel" style="height:0;overflow:hidden;"></div>',
                unsafe_allow_html=True)
    # 4 real st.button() columns — CSS/JS will fix this row to the bottom
    items = [("📤", "Upload", 0), ("📷", "Live", 1), ("📊", "Stats", 2), ("🕒", "History", 3)]
    cols  = st.columns(4)
    for col, (icon, name, idx) in zip(cols, items):
        # Wrap active tab in a div so CSS can target it
        active = (tab == idx)
        if active:
            st.markdown(f'<style>.bnav-active-{idx} .stButton>button{{color:#e24b4a!important;}}</style>', unsafe_allow_html=True)
        with col:
            label = f"{icon} {name}"
            if st.button(label, key=f"bnav_{idx}", type="secondary"):
                st.session_state.tab = idx; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
def render_upload_tab(model, mok, mk, mk_label, labels):
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="mob-page-title">📤 Upload · {mk_label}</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload", type=["jpg","jpeg","png","webp","bmp","tiff","tif"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    if not uploaded_files:
        st.markdown("""
<div class="dz">
  <div class="dz-icon">🚦</div>
  <div class="dz-title">Tap to upload a sign photo</div>
  <div class="dz-sub">Up to 5 images · PNG JPG WEBP BMP TIFF</div>
</div>""", unsafe_allow_html=True)
    else:
        files = uploaded_files[:5]
        from PIL import Image
        if len(files) == 1:
            f = files[0]
            raw = Image.open(f)
            proc, warns, changed = preprocess_image(raw)
            if changed:
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
                st.markdown(f'<div class="ptime">⚡ {ms:.0f} ms</div>', unsafe_allow_html=True)
                if probs[0] < NO_SIGN_THRESHOLD:
                    render_no_sign(probs[0])
                else:
                    render_pred(ids[0], probs[0], mk, mk_label, labels)
                    render_top3(ids, probs, labels, mk)
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    st.session_state.upload_history.insert(0, {
                        "name": labels.get(str(ids[0]), f"Class {ids[0]}")[:30],
                        "conf": round(probs[0]*100,1),
                        "danger": get_info(mk, int(ids[0]))["danger"],
                        "model": mk, "time": ts,
                        "top2": labels.get(str(ids[1]),""),
                        "top3": labels.get(str(ids[2]),"")
                    })
                    st.session_state.upload_history = st.session_state.upload_history[:MAX_HISTORY]
        else:
            # Batch: single column on mobile
            for i, f in enumerate(files):
                raw = Image.open(f); proc, warns, _ = preprocess_image(raw)
                st.image(proc, use_column_width=True, caption=f.name[:20])
                if warns: render_warns(warns[:1])
                if mok:
                    with st.spinner(""):
                        ids, probs, ms = predict(model, proc)
                    if probs[0] < NO_SIGN_THRESHOLD:
                        st.markdown('<div style="font-size:12px;color:#4b5563;text-align:center;padding:6px;">🔍 No sign detected</div>', unsafe_allow_html=True)
                    else:
                        name = labels.get(str(ids[0]), f"Class {ids[0]}")
                        info = get_info(mk, int(ids[0]))
                        color= DANGER_COLORS.get(info["danger"],"#22c55e")
                        icon = DANGER_ICONS.get(info["danger"],"🟢")
                        st.markdown(f'<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:8px;padding:10px;margin-bottom:10px;text-align:center;"><div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">{name[:24]}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:22px;font-weight:700;color:{color};">{probs[0]*100:.1f}%</div><div style="font-size:10px;margin-top:4px;">{icon} {info["danger"]}</div></div>', unsafe_allow_html=True)
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.upload_history.insert(0, {
                            "name": name[:30],"conf":round(probs[0]*100,1),
                            "danger":info["danger"],"model":mk,"time":ts,
                            "top2":labels.get(str(ids[1]),""),"top3":labels.get(str(ids[2]),"")
                        })
            st.session_state.upload_history = st.session_state.upload_history[:MAX_HISTORY]

    # Quick stats below upload on mobile
    n_up  = len(st.session_state.upload_history)
    avg_c = np.mean([h["conf"] for h in st.session_state.upload_history]) if st.session_state.upload_history else 0.0
    from shared import load_summary
    summary = load_summary()
    i_sum = summary.get(f"{mk}_model", summary)
    t1 = i_sum.get("final_val_acc", 76.81 if mk=="indian" else 83.42)
    t3 = i_sum.get("final_top3_acc", 92.80 if mk=="indian" else 97.09)
    st.markdown(f"""<div class="sg" style="margin-top:14px;">
  <div class="sc"><div class="sl">Top-1 Acc</div><div class="sv" style="color:#22c55e;">{t1:.1f}%</div></div>
  <div class="sc"><div class="sl">Top-3 Acc</div><div class="sv" style="color:#3b82f6;">{t3:.1f}%</div></div>
  <div class="sc"><div class="sl">Classified</div><div class="sv">{n_up}</div></div>
  <div class="sc"><div class="sl">Avg Conf</div><div class="sv" style="color:#e24b4a;">{avg_c:.1f}%</div></div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_live_tab(model, mok, mk, mk_label, labels):
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="mob-page-title">📷 Live · {mk_label}</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:10px;padding:8px 14px;margin-bottom:10px;font-size:11px;color:#4b5563;">🎯 Centre the sign · fill 60%+ of frame · hold steady for 2 seconds 📱</div>', unsafe_allow_html=True)
    cam = st.camera_input("Camera", label_visibility="collapsed", key=f"mob_cam_{mk}")
    ca1, ca2 = st.columns(2)
    with ca1: auto = st.checkbox("🔄 Auto-refresh", value=not st.session_state.freeze)
    with ca2:
        if st.button("❄️ Freeze" if not st.session_state.freeze else "▶️ Resume", key="mob_frz", type="secondary"):
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
        st.markdown(f'<div class="ptime">⚡ {r["ms"]:.0f} ms{"  · ❄️" if st.session_state.freeze else ""}</div>', unsafe_allow_html=True)
        if r["no_sign"]:
            st.markdown(f'<div style="background:#0d0f18;border:1px dashed #1a1d2e;border-radius:12px;padding:24px 16px;text-align:center;margin-bottom:10px;"><div style="font-size:28px;margin-bottom:8px;">🔍</div><div style="font-size:14px;font-weight:600;color:#4b5563;">No Sign Detected</div><div style="font-size:11px;color:#374151;margin-top:6px;">{r["conf"]*100:.1f}% — Centre sign &amp; hold steady</div></div>', unsafe_allow_html=True)
        else:
            cid  = r["cid"]; conf = r["conf"]
            name = labels.get(str(cid), f"Class {cid}")
            info = get_info(mk, cid); color = DANGER_COLORS.get(info["danger"],"#22c55e")
            st.markdown(f"""<div class="card"><div class="ct"></div><div class="card-body">
  <div style="font-size:9px;color:#4b5563;letter-spacing:1px;margin-bottom:4px;">DETECTED SIGN</div>
  <div class="live-name">{name}</div>
  <div style="margin-top:6px;margin-bottom:10px;">{badge(info['danger'])}</div>
  {conf_ind(conf)}
  <div class="live-conf" style="color:{color};">{conf*100:.1f}%</div>
  <div class="abox" style="margin-top:0;">
    <div class="blabel" style="color:#22c55e;">ADAS Action</div>
    <div class="btext" style="color:#86efac;">{info['action']}</div>
  </div>
</div></div>""", unsafe_allow_html=True)
            render_top3(r["ids"], r["probs"], labels, mk)
    if st.session_state.sparkline:
        st.markdown('<div class="slabel">Confidence Trend</div>', unsafe_allow_html=True)
        sp = draw_sparkline(st.session_state.sparkline)
        if sp: st.image(sp, use_column_width=True)
    # Session stats
    wh = st.session_state.webcam_history; n_det = len(wh)
    avg_c = np.mean([h["conf"] for h in wh]) if wh else 0
    st.markdown(f'<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:10px;padding:12px 14px;margin-top:10px;display:flex;justify-content:space-between;align-items:center;"><span style="font-size:11px;color:#4b5563;">Signs detected</span><span style="font-family:\'JetBrains Mono\',monospace;font-size:22px;font-weight:700;color:#22c55e;">{n_det}</span></div>', unsafe_allow_html=True)
    if st.button("🗑 Clear Webcam History", key="mob_cwh", type="secondary"):
        st.session_state.webcam_history=[]; st.session_state.live_result=None; st.session_state.sparkline=[]; st.rerun()
    if auto and cam and not st.session_state.freeze:
        time.sleep(0.35); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_stats_tab(mk):
    from shared import load_summary
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown('<div class="mob-page-title">📊 Model Stats</div>', unsafe_allow_html=True)
    summary = load_summary()
    i_s = summary.get("indian_model",{}); g_s = summary.get("gtsrb_model",{})
    i_t1=i_s.get("final_val_acc",76.81); i_t3=i_s.get("final_top3_acc",92.80)
    i_p1=i_s.get("phase1_best_val_acc",49.19); i_p2=i_s.get("phase2_best_val_acc",76.81)
    g_t1=g_s.get("final_val_acc",83.42); g_t3=g_s.get("final_top3_acc",97.09)
    g_p1=g_s.get("phase1_best_val_acc",39.53); g_p2=g_s.get("phase2_best_val_acc",83.42)
    # Gauges side by side
    gc1, gc2 = st.columns(2)
    with gc1:
        g1 = draw_gauge(i_t1,"#93c5fd","🇮🇳 Indian")
        if g1: st.image(g1, use_column_width=True)
    with gc2:
        g2 = draw_gauge(g_t1,"#fca5a5","🌍 GTSRB")
        if g2: st.image(g2, use_column_width=True)
    # Comparison table
    comp = [
        ("Top-1 Acc",f"{i_t1:.1f}%",f"{g_t1:.1f}%"),
        ("Top-3 Acc",f"{i_t3:.1f}%",f"{g_t3:.1f}%"),
        ("Phase 1",f"{i_p1:.1f}%",f"{g_p1:.1f}%"),
        ("Phase 2",f"{i_p2:.1f}%",f"{g_p2:.1f}%"),
        ("Classes","59","43"),
    ]
    rows_h = "".join(f'<tr><td class="met">{m}</td><td class="vv" style="color:#93c5fd;">{iv}</td><td class="vv" style="color:#fca5a5;">{gv}</td></tr>' for m,iv,gv in comp)
    st.markdown(f'<div class="card" style="margin-top:12px;"><div class="card-body"><table class="comptbl"><thead><tr><th>Metric</th><th style="text-align:center;">🇮🇳</th><th style="text-align:center;">🌍</th></tr></thead><tbody>{rows_h}</tbody></table></div></div>', unsafe_allow_html=True)
    # Training curve (compact)
    ih1=i_s.get("training_history",{}).get("phase1",{}); ih2=i_s.get("training_history",{}).get("phase2",{})
    gh1=g_s.get("training_history",{}).get("phase1",{}); gh2=g_s.get("training_history",{}).get("phase2",{})
    i_val=ih1.get("val_acc",[])+ih2.get("val_acc",[]); g_val=gh1.get("val_acc",[])+gh2.get("val_acc",[])
    if not i_val: i_val=[28,31,37,35,41,37,41,43,41,45,45,45,43,42,46,45,48,49,48,49,49,48,47,48,47,57,61,65,67,70,69,71,71,73,74,75,75,75,76,76,76,76,76]
    if not g_val: g_val=[19,25,30,28,34,30,34,34,33,36,34,36,37,37,37,38,38,38,38,38,38,38,39,39,39,49,54,58,61,65,68,71,73,75,77,79,80,81,81,82,81,82,83]
    fig,ax=plt.subplots(figsize=(5,2.5)); fig.patch.set_facecolor("#0d0f18"); ax.set_facecolor("#080a10")
    ax.plot(range(1,len(i_val)+1),i_val,color="#93c5fd",lw=1.5,label="🇮🇳")
    ax.plot(range(1,len(g_val)+1),g_val,color="#fca5a5",lw=1.5,label="🌍")
    ax.set_xlabel("Epoch",fontsize=8,color="#4b5563"); ax.set_ylabel("Val Acc (%)",fontsize=8,color="#4b5563")
    ax.tick_params(colors="#4b5563",labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#1a1d2e")
    ax.grid(True,color="#1a1d2e",alpha=0.7,lw=0.5); ax.legend(fontsize=8,labelcolor="#d1d5db",facecolor="#0d0f18",edgecolor="#1a1d2e")
    plt.tight_layout(); buf=_io.BytesIO(); plt.savefig(buf,format="png",dpi=110,bbox_inches="tight",facecolor="#0d0f18"); buf.seek(0); plt.close()
    st.markdown('<div class="slabel" style="margin-top:10px;">Training Curves</div>', unsafe_allow_html=True)
    st.image(buf, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_history_tab():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown('<div class="mob-page-title">🕒 History</div>', unsafe_allow_html=True)
    uh = st.session_state.upload_history
    wh = st.session_state.webcam_history
    all_h = uh + wh
    dc = {"MANDATORY":"#fca5a5","WARNING":"#fcd34d","INFO":"#86efac"}
    if all_h:
        counts   = Counter(h["name"] for h in all_h)
        top_sign, top_count = counts.most_common(1)[0]
        top_info = next((h for h in all_h if h["name"] == top_sign), {})
        icon  = DANGER_ICONS.get(top_info.get("danger","INFO"),"🟢")
        color = DANGER_COLORS.get(top_info.get("danger","INFO"),"#22c55e")
        st.markdown(f'<div class="most-det"><div class="most-det-icon">{icon}</div><div><div class="most-det-label">Most Detected</div><div class="most-det-name">{top_sign[:22]}</div></div><div class="most-det-count" style="color:{color};">{top_count}×</div></div>', unsafe_allow_html=True)
    # Upload history
    st.markdown('<div class="slabel">📤 Upload History</div>', unsafe_allow_html=True)
    if uh:
        su = st.text_input("Filter signs…", key="mob_su", label_visibility="collapsed", placeholder="Filter by name…")
        fu = [h for h in uh if su.lower() in h["name"].lower()] if su else uh
        st.markdown('<div class="card"><div class="card-body"><table class="htbl"><thead><tr><th>Sign</th><th>Conf</th><th>Type</th></tr></thead><tbody>', unsafe_allow_html=True)
        for h in fu[:30]:
            c_ = dc.get(h["danger"],"#86efac"); ico = DANGER_ICONS.get(h["danger"],"🟢")
            st.markdown(f'<tr><td class="hname">{h["name"][:18]}</td><td class="hconf" style="color:{c_};">{h["conf"]:.0f}%</td><td>{ico}</td></tr>', unsafe_allow_html=True)
        st.markdown('</tbody></table></div></div>', unsafe_allow_html=True)
        csv_u = make_csv([[h["name"],h["conf"],h["danger"],h["model"],h["time"],h.get("top2",""),h.get("top3","")] for h in uh], ["Sign","Confidence%","Type","Model","Time","Top2","Top3"])
        st.download_button("⬇️ Export CSV", csv_u, "upload_history.csv","text/csv",key="mob_dlu")
    else:
        st.markdown('<div style="text-align:center;padding:20px;color:#374151;font-size:12px;">No upload history yet</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    # Webcam history
    st.markdown('<div class="slabel">📷 Webcam History</div>', unsafe_allow_html=True)
    if wh:
        st.markdown('<div class="card"><div class="card-body"><table class="htbl"><thead><tr><th>Sign</th><th>Conf</th><th>Type</th></tr></thead><tbody>', unsafe_allow_html=True)
        for h in wh[:30]:
            c_ = dc.get(h["danger"],"#86efac"); ico = DANGER_ICONS.get(h["danger"],"🟢")
            st.markdown(f'<tr><td class="hname">{h["name"][:18]}</td><td class="hconf" style="color:{c_};">{h["conf"]:.0f}%</td><td>{ico}</td></tr>', unsafe_allow_html=True)
        st.markdown('</tbody></table></div></div>', unsafe_allow_html=True)
        csv_w = make_csv([[h["name"],h["conf"],h["danger"],h["model"],h["time"]] for h in wh], ["Sign","Confidence%","Type","Model","Time"])
        st.download_button("⬇️ Export CSV", csv_w, "webcam_history.csv","text/csv",key="mob_dlw")
        if st.button("🗑 Clear All History", key="mob_clrall", type="secondary"):
            st.session_state.upload_history=[]
            st.session_state.webcam_history=[]
            st.session_state.live_result=None
            st.session_state.sparkline=[]
            st.rerun()
    else:
        st.markdown('<div style="text-align:center;padding:20px;color:#374151;font-size:12px;">No webcam detections yet</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
