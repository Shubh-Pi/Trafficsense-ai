"""
mobile_ui.py — Mobile layout for TrafficSense AI

ARCHITECTURE: The entire mobile UI is rendered as a single HTML page via
st.components.v1.html(). This completely bypasses Streamlit's layout engine
so st.columns() stacking issues on HF Spaces never occur.

Communication back to Python:
  - Tab/model switching: window.parent.postMessage({type:'setTab', val:N})
    app.py reads st.query_params which are set by a tiny JS postMessage bridge.
  - File upload: rendered inside the HTML component with a real <input type=file>
    The file is base64-encoded and sent to the Anthropic-style predict endpoint
    via a hidden st.file_uploader that's synced.

Simpler approach used here: HTML nav + model row are pure HTML inside the
component. For actions that need Python (upload, predict), we fall back to
st.file_uploader rendered BELOW the HTML shell, hidden under the nav.
"""

import io as _io, datetime, time, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from collections import Counter
from PIL import Image

from shared import (
    DANGER_ICONS, DANGER_COLORS, NO_SIGN_THRESHOLD, MAX_HISTORY,
    get_info, preprocess_image, predict, make_csv,
    badge, conf_ind, render_warns, render_pred, render_no_sign,
    render_top3, draw_sparkline, draw_gauge, BASE_CSS
)

MOBILE_CSS = ""  # not used — HTML shell has its own styles


def _nav_html(mk, tab, both_ok):
    status_color = "#22c55e" if both_ok else "#f59e0b"
    status_text  = "Ready"  if both_ok else "Partial"
    i_act = "active" if mk == "indian" else ""
    g_act = "active" if mk == "gtsrb"  else ""
    tabs  = [("📤","Upload",0),("📷","Live",1),("📊","Stats",2),("🕒","History",3)]
    bnav  = ""
    for idx, icon, name in tabs:
        cls = "active" if tab == idx else ""
        bnav += f'<a href="?tab={idx}&mk={mk}" class="bn-item {cls}"><span class="bn-icon">{icon}</span><span>{name}</span></a>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#080a10;font-family:'Inter',system-ui,sans-serif;color:#f0f0f0;}}
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  .navbar{{
    background:#0d0f18;
    border-bottom:1px solid #1a1d2e;
    padding:0 14px;
  }}
  .nav-top{{
    display:flex;
    align-items:center;
    justify-content:space-between;
    height:48px;
  }}
  .nav-logo{{display:flex;align-items:center;gap:10px;}}
  .nav-icon{{width:34px;height:34px;background:rgba(226,75,74,0.15);border:1px solid rgba(226,75,74,0.35);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;}}
  .nav-brand{{font-size:16px;font-weight:700;color:#f0f0f0;letter-spacing:-0.3px;}}
  .nav-brand span{{color:#e24b4a;}}
  .status-pill{{display:flex;align-items:center;gap:5px;background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:20px;padding:4px 10px;font-size:11px;color:{status_color};font-weight:500;}}
  .sdot{{width:6px;height:6px;border-radius:50%;background:{status_color};}}

  .model-row{{
    display:flex;
    flex-direction:row;
    gap:8px;
    padding:8px 0 10px;
  }}
  .model-row a{{
    flex:1;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:8px 6px;
    border-radius:9px;
    font-size:13px;
    font-weight:600;
    text-decoration:none;
    border:1px solid #1a1d2e;
    background:#13151f;
    color:#6b7280;
    white-space:nowrap;
  }}
  .model-row a.active{{
    background:#e24b4a;
    color:#fff;
    border-color:#e24b4a;
    box-shadow:0 2px 8px rgba(226,75,74,0.3);
  }}

  .bnav{{
    position:fixed;
    bottom:0;left:0;right:0;
    background:#0d0f18;
    border-top:1px solid #1a1d2e;
    display:flex;
    flex-direction:row;
    z-index:9999;
    padding-bottom:env(safe-area-inset-bottom,0px);
  }}
  .bn-item{{
    flex:1;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    padding:8px 2px 6px;
    text-decoration:none;
    color:#4b5563;
    font-size:10px;
    font-weight:500;
    border-top:2px solid transparent;
    gap:3px;
    min-height:50px;
  }}
  .bn-icon{{font-size:19px;line-height:1;}}
  .bn-item.active{{color:#e24b4a;border-top:2px solid #e24b4a;}}
</style>
</head>
<body>
<div class="navbar">
  <div class="nav-top">
    <div class="nav-logo">
      <div class="nav-icon">🚦</div>
      <div class="nav-brand">TrafficSense<span>AI</span></div>
    </div>
    <div class="status-pill">
      <div class="sdot"></div>
      <span>{status_text}</span>
    </div>
  </div>
  <div class="model-row">
    <a href="?tab={tab}&mk=indian" class="{i_act}">🇮🇳 Indian</a>
    <a href="?tab={tab}&mk=gtsrb"  class="{g_act}">🌍 GTSRB</a>
  </div>
</div>
<div class="bnav">{bnav}</div>
</body>
</html>"""


def render_navbar(mk, both_ok):
    tab = st.session_state.get("tab", 0)
    # Render the nav shell as a fixed-height HTML component
    components.html(_nav_html(mk, tab, both_ok), height=110, scrolling=False)


def render_model_strip(mk):
    pass


def render_bottom_nav(tab):
    mk = st.session_state.get("mk", "gtsrb")
    tabs  = [("📤","Upload",0),("📷","Live",1),("📊","Stats",2),("🕒","History",3)]
    bnav  = ""
    for idx, icon, name in tabs:
        cls = "active" if tab == idx else ""
        bnav += f'<a href="?tab={idx}&mk={mk}" class="bn-item {cls}"><span class="bn-icon">{icon}</span><span>{name}</span></a>'

    html = f"""<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#0d0f18;font-family:system-ui,sans-serif;}}
.bnav{{
  display:flex;flex-direction:row;
  background:#0d0f18;
  border-top:1px solid #1a1d2e;
  width:100%;
}}
.bn-item{{
  flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  padding:8px 2px 6px;
  text-decoration:none;
  color:#4b5563;font-size:10px;font-weight:500;
  border-top:2px solid transparent;
  gap:3px;min-height:50px;
}}
.bn-icon{{font-size:19px;line-height:1;}}
.bn-item.active{{color:#e24b4a;border-top:2px solid #e24b4a;}}
</style></head>
<body><div class="bnav">{bnav}</div></body></html>"""
    components.html(html, height=58, scrolling=False)


# ── Shared page CSS injected once for Streamlit widgets below the nav ──────────
_PAGE_CSS = BASE_CSS + """
<style>
.wrap{padding:8px 12px 70px 12px !important;}
.mob-page-title{font-size:10px;font-weight:700;color:#374151;letter-spacing:1.5px;text-transform:uppercase;padding:6px 0 5px;margin-bottom:8px;border-bottom:1px solid #1a1d2e;}
.pred-name{font-size:16px !important;}
.conf-big {font-size:34px !important;}
.live-conf{font-size:42px !important;}
.live-name{font-size:16px !important;}
.sv       {font-size:22px !important;}
.bn       {width:110px !important;}
.status-pill{padding:4px 10px !important;font-size:10px !important;}
</style>
"""


def _inject_css():
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_upload_tab(model, mok, mk, mk_label, labels):
    _inject_css()
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
            for f in files:
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
                        st.markdown(f'<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:8px;padding:10px;margin-bottom:10px;text-align:center;"><div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">{name[:24]}</div><div style="font-family:monospace;font-size:22px;font-weight:700;color:{color};">{probs[0]*100:.1f}%</div><div style="font-size:10px;margin-top:4px;">{icon} {info["danger"]}</div></div>', unsafe_allow_html=True)
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.upload_history.insert(0, {
                            "name": name[:30],"conf":round(probs[0]*100,1),
                            "danger":info["danger"],"model":mk,"time":ts,
                            "top2":labels.get(str(ids[1]),""),"top3":labels.get(str(ids[2]),"")
                        })
            st.session_state.upload_history = st.session_state.upload_history[:MAX_HISTORY]

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
    _inject_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="mob-page-title">📷 Live · {mk_label}</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:10px;padding:8px 14px;margin-bottom:10px;font-size:11px;color:#4b5563;">🎯 Centre the sign · fill 60%+ of frame · hold steady 📱</div>', unsafe_allow_html=True)
    cam = st.camera_input("Camera", label_visibility="collapsed", key=f"mob_cam_{mk}")
    ca1, ca2 = st.columns(2)
    with ca1: auto = st.checkbox("🔄 Auto-refresh", value=not st.session_state.freeze)
    with ca2:
        if st.button("❄️ Freeze" if not st.session_state.freeze else "▶️ Resume", key="mob_frz", type="secondary"):
            st.session_state.freeze = not st.session_state.freeze; st.rerun()
    if cam and mok and not st.session_state.freeze:
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
    wh = st.session_state.webcam_history; n_det = len(wh)
    st.markdown(f'<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:10px;padding:12px 14px;margin-top:10px;display:flex;justify-content:space-between;align-items:center;"><span style="font-size:11px;color:#4b5563;">Signs detected</span><span style="font-family:monospace;font-size:22px;font-weight:700;color:#22c55e;">{n_det}</span></div>', unsafe_allow_html=True)
    if st.button("🗑 Clear Webcam History", key="mob_cwh", type="secondary"):
        st.session_state.webcam_history=[]; st.session_state.live_result=None; st.session_state.sparkline=[]; st.rerun()
    if auto and cam and not st.session_state.freeze:
        time.sleep(0.35); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
def render_stats_tab(mk):
    from shared import load_summary
    _inject_css()
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown('<div class="mob-page-title">📊 Model Stats</div>', unsafe_allow_html=True)
    summary = load_summary()
    i_s = summary.get("indian_model",{}); g_s = summary.get("gtsrb_model",{})
    i_t1=i_s.get("final_val_acc",76.81); i_t3=i_s.get("final_top3_acc",92.80)
    i_p1=i_s.get("phase1_best_val_acc",49.19); i_p2=i_s.get("phase2_best_val_acc",76.81)
    g_t1=g_s.get("final_val_acc",83.42); g_t3=g_s.get("final_top3_acc",97.09)
    g_p1=g_s.get("phase1_best_val_acc",39.53); g_p2=g_s.get("phase2_best_val_acc",83.42)
    gc1, gc2 = st.columns(2)
    with gc1:
        g1 = draw_gauge(i_t1,"#93c5fd","🇮🇳 Indian")
        if g1: st.image(g1, use_column_width=True)
    with gc2:
        g2 = draw_gauge(g_t1,"#fca5a5","🌍 GTSRB")
        if g2: st.image(g2, use_column_width=True)
    comp = [
        ("Top-1 Acc",f"{i_t1:.1f}%",f"{g_t1:.1f}%"),
        ("Top-3 Acc",f"{i_t3:.1f}%",f"{g_t3:.1f}%"),
        ("Phase 1",f"{i_p1:.1f}%",f"{g_p1:.1f}%"),
        ("Phase 2",f"{i_p2:.1f}%",f"{g_p2:.1f}%"),
        ("Classes","59","43"),
    ]
    rows_h = "".join(f'<tr><td class="met">{m}</td><td class="vv" style="color:#93c5fd;">{iv}</td><td class="vv" style="color:#fca5a5;">{gv}</td></tr>' for m,iv,gv in comp)
    st.markdown(f'<div class="card" style="margin-top:12px;"><div class="card-body"><table class="comptbl"><thead><tr><th>Metric</th><th style="text-align:center;">🇮🇳</th><th style="text-align:center;">🌍</th></tr></thead><tbody>{rows_h}</tbody></table></div></div>', unsafe_allow_html=True)
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
    _inject_css()
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