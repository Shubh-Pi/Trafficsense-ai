import os
import io
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3

st.set_page_config(page_title="TrafficSense AI", page_icon="🚦", layout="wide", initial_sidebar_state="collapsed")

IMG_SIZE = 224
DEVICE   = torch.device("cpu")
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
NO_SIGN_THRESHOLD = 0.30

class TrafficSignNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = efficientnet_b3(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_features, 512), nn.SiLU(),
            nn.BatchNorm1d(512), nn.Dropout(p=0.3), nn.Linear(512, 256),
            nn.SiLU(), nn.BatchNorm1d(256), nn.Dropout(p=0.2), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.base(x)

@st.cache_resource
def load_models():
    results = {}
    for key, fname, n in [("indian","indian_traffic_sign_model.pth",59),("gtsrb","gtsrb_traffic_sign_model.pth",43)]:
        model = TrafficSignNet(num_classes=n)
        loaded = False
        for path in [fname, f"/app/{fname}", os.path.join(os.path.dirname(__file__), fname)]:
            if not os.path.exists(path): continue
            try:
                obj = torch.load(path, map_location=DEVICE, weights_only=False)
                state = obj if isinstance(obj, dict) else obj.state_dict()
                missing, _ = model.load_state_dict(state, strict=False)
                if 1 - len(missing)/len(model.state_dict()) >= 0.85:
                    model.eval(); loaded = True; break
            except: pass
        results[key] = (model, loaded)
    return results

@st.cache_data
def load_all_labels():
    out = {}
    for key, fname, n in [("indian","indian_labels.json",59),("gtsrb","gtsrb_labels.json",43)]:
        for p in [fname, f"/app/{fname}", os.path.join(os.path.dirname(__file__), fname)]:
            if os.path.exists(p):
                with open(p) as f: out[key] = json.load(f); break
        if key not in out: out[key] = {str(i): f"Class {i}" for i in range(n)}
    return out

@st.cache_data
def load_summary():
    for f in ["training_summary.json", "/app/training_summary.json"]:
        if os.path.exists(f):
            with open(f) as fh: return json.load(fh)
    return {}

GTSRB_INFO = {
    "0":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 20 km/h.","action":"Reduce to 20 km/h immediately."},
    "1":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 30 km/h — residential zone.","action":"Reduce to 30 km/h. Watch for pedestrians."},
    "2":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 50 km/h — standard urban limit.","action":"Maintain at or below 50 km/h."},
    "3":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 60 km/h.","action":"Reduce to 60 km/h."},
    "4":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 70 km/h.","action":"Reduce to 70 km/h."},
    "5":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 80 km/h.","action":"Reduce to 80 km/h."},
    "6":{"danger":"INFO","color":"#22c55e","meaning":"End of 80 km/h restriction.","action":"Drive to conditions."},
    "7":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 100 km/h.","action":"Do not exceed 100 km/h."},
    "8":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Speed limit 120 km/h.","action":"Do not exceed 120 km/h."},
    "9":{"danger":"MANDATORY","color":"#e24b4a","meaning":"No overtaking for all vehicles.","action":"Stay in lane. Do not overtake."},
    "10":{"danger":"MANDATORY","color":"#e24b4a","meaning":"No overtaking for vehicles over 3.5t.","action":"Heavy vehicles must not overtake."},
    "11":{"danger":"INFO","color":"#22c55e","meaning":"Right of way at next intersection.","action":"You have priority. Maintain speed."},
    "12":{"danger":"INFO","color":"#22c55e","meaning":"Priority road — continuous right of way.","action":"Maintain speed. You have priority."},
    "13":{"danger":"WARNING","color":"#f59e0b","meaning":"Yield — give way to main road traffic.","action":"Slow down and yield to all traffic."},
    "14":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Stop sign — complete stop required.","action":"Stop completely before the line."},
    "15":{"danger":"MANDATORY","color":"#e24b4a","meaning":"No vehicles permitted.","action":"Do not enter. Find alternative route."},
    "16":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Vehicles over 3.5t prohibited.","action":"Heavy vehicles must use alternative route."},
    "17":{"danger":"MANDATORY","color":"#e24b4a","meaning":"No entry — one-way road.","action":"Do not enter. Turn around."},
    "18":{"danger":"WARNING","color":"#f59e0b","meaning":"General hazard warning ahead.","action":"Reduce speed. Be alert."},
    "19":{"danger":"WARNING","color":"#f59e0b","meaning":"Dangerous left curve ahead.","action":"Reduce speed before the curve."},
    "20":{"danger":"WARNING","color":"#f59e0b","meaning":"Dangerous right curve ahead.","action":"Reduce speed before the curve."},
    "21":{"danger":"WARNING","color":"#f59e0b","meaning":"Double curve ahead.","action":"Reduce speed. Keep to lane."},
    "22":{"danger":"WARNING","color":"#f59e0b","meaning":"Bumpy road ahead.","action":"Reduce speed to protect suspension."},
    "23":{"danger":"WARNING","color":"#f59e0b","meaning":"Slippery road ahead.","action":"Reduce speed. Avoid sudden inputs."},
    "24":{"danger":"WARNING","color":"#f59e0b","meaning":"Road narrows on the right.","action":"Move centre-left. Watch oncoming traffic."},
    "25":{"danger":"WARNING","color":"#f59e0b","meaning":"Roadworks ahead.","action":"Reduce speed. Watch for workers."},
    "26":{"danger":"WARNING","color":"#f59e0b","meaning":"Traffic signals ahead.","action":"Prepare to stop at red light."},
    "27":{"danger":"WARNING","color":"#f59e0b","meaning":"Pedestrian crossing ahead.","action":"Slow down. Yield to pedestrians."},
    "28":{"danger":"WARNING","color":"#f59e0b","meaning":"Children crossing — school zone.","action":"Reduce speed sharply. Watch for children."},
    "29":{"danger":"WARNING","color":"#f59e0b","meaning":"Bicycle crossing ahead.","action":"Check mirrors. Give cyclists space."},
    "30":{"danger":"WARNING","color":"#f59e0b","meaning":"Ice or snow likely.","action":"Reduce speed. Increase following distance."},
    "31":{"danger":"WARNING","color":"#f59e0b","meaning":"Wild animals crossing ahead.","action":"Reduce speed. Scan road edges."},
    "32":{"danger":"INFO","color":"#22c55e","meaning":"End of all restrictions.","action":"Normal rules apply. Drive to conditions."},
    "33":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Turn right — mandatory.","action":"Signal right. Prepare to turn right."},
    "34":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Turn left — mandatory.","action":"Signal left. Prepare to turn left."},
    "35":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Proceed straight only.","action":"Do not turn. Go straight only."},
    "36":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Go straight or turn right only.","action":"Left turns not permitted."},
    "37":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Go straight or turn left only.","action":"Right turns not permitted."},
    "38":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Keep right.","action":"Pass obstacle on the right side."},
    "39":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Keep left.","action":"Pass obstacle on the left side."},
    "40":{"danger":"MANDATORY","color":"#e24b4a","meaning":"Roundabout — yield to circulating traffic.","action":"Yield before entering roundabout."},
    "41":{"danger":"INFO","color":"#22c55e","meaning":"End of no-overtaking zone.","action":"Overtaking now permitted where safe."},
    "42":{"danger":"INFO","color":"#22c55e","meaning":"End of no-overtaking zone for heavy vehicles.","action":"Heavy vehicles may overtake where safe."},
}
INDIAN_INFO_DEFAULT = {"danger":"INFO","color":"#3b82f6","meaning":"Indian road sign detected.","action":"Follow standard road rules and local traffic laws."}

def get_sign_info(mk, cid):
    if mk == "gtsrb": return GTSRB_INFO.get(str(cid), {"danger":"INFO","color":"#22c55e","meaning":"No description.","action":"Follow standard rules."})
    return INDIAN_INFO_DEFAULT

val_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

def flatten_alpha(img):
    if img.mode in ("RGBA","LA") or (img.mode=="P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255,255,255))
        if img.mode == "P": img = img.convert("RGBA")
        mask = img.split()[-1] if img.mode=="RGBA" else None
        bg.paste(img.convert("RGB"), mask=mask); return bg
    return img.convert("RGB")

def predict(model, img_pil):
    t0 = time.time()
    tensor = val_transform(flatten_alpha(img_pil)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    ms = (time.time()-t0)*1000
    topk_probs, topk_ids = probs.topk(5)
    return topk_ids.cpu().numpy(), topk_probs.cpu().numpy(), ms

# ── CSS (v0 design translated) ────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0c0d12 !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Navbar ── */
.navbar {
    background: #13141c;
    border-bottom: 1px solid #1e2030;
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
    position: sticky; top: 0; z-index: 100;
}
.nav-logo { display: flex; align-items: center; gap: 12px; }
.nav-logo-icon {
    width: 36px; height: 36px;
    background: rgba(226,75,74,0.15);
    border: 1px solid rgba(226,75,74,0.4);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.nav-brand { font-size: 16px; font-weight: 700; color: #f0f0f0; }
.nav-brand span { color: #e24b4a; }
.nav-sub { font-size: 10px; color: #6b7280; margin-top: 1px; }
.nav-right { display: flex; align-items: center; gap: 10px; }
.model-switcher { display: flex; background: #1e2030; border-radius: 8px; padding: 3px; gap: 2px; }
.model-btn {
    padding: 5px 14px; border-radius: 6px; font-size: 12px; font-weight: 500;
    border: none; cursor: pointer; transition: all 0.15s;
    color: #6b7280; background: transparent; font-family: 'Inter', sans-serif;
}
.model-btn.active { background: #e24b4a; color: white; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px; padding: 5px 12px;
    font-size: 11px; color: #22c55e; font-weight: 500;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Tabs ── */
.tabs-bar {
    display: flex; gap: 2px;
    background: rgba(30,32,48,0.5);
    border: 1px solid #1e2030;
    border-radius: 8px; padding: 3px;
    margin: 16px 24px 0;
}
.tab-btn-v {
    flex: 1; padding: 8px 12px;
    font-size: 12px; font-weight: 500;
    border: none; border-radius: 6px;
    cursor: pointer; transition: all 0.15s;
    color: #6b7280; background: transparent;
    font-family: 'Inter', sans-serif;
    display: flex; align-items: center; justify-content: center; gap: 6px;
}
.tab-btn-v.active { background: #e24b4a; color: white; }
.tab-btn-v:hover:not(.active) { color: #d1d5db; }

/* ── Content ── */
.content { padding: 20px 24px; }

/* ── Cards ── */
.ts-card {
    background: #13141c;
    border: 1px solid #1e2030;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 14px;
}
.ts-card-body { padding: 16px 18px; }
.ts-card-accent { height: 2px; background: linear-gradient(90deg, #e24b4a, rgba(226,75,74,0.2), transparent); }

/* ── Upload zone ── */
.dropzone {
    border: 2px dashed #1e2030;
    border-radius: 10px;
    padding: 48px 20px;
    text-align: center;
    background: #0c0d12;
    cursor: pointer;
    transition: border-color 0.2s;
    margin-bottom: 14px;
}
.dropzone:hover { border-color: rgba(226,75,74,0.4); }
.dropzone-icon { font-size: 40px; margin-bottom: 12px; }
.dropzone-title { font-size: 14px; font-weight: 500; color: #f0f0f0; margin-bottom: 4px; }
.dropzone-sub   { font-size: 12px; color: #6b7280; }

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.stat-card { background: #13141c; border: 1px solid #1e2030; border-radius: 10px; padding: 14px 16px; }
.stat-label { font-size: 9px; font-weight: 600; color: #6b7280; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.stat-value { font-family: 'Geist Mono', monospace; font-size: 26px; font-weight: 700; color: #f0f0f0; }

/* ── Prediction card ── */
.pred-topbar { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 14px; }
.pred-name { font-size: 20px; font-weight: 700; color: #f0f0f0; line-height: 1.2; }
.pred-sub  { font-size: 11px; color: #6b7280; margin-top: 3px; }
.pred-conf-big { font-family: 'Geist Mono', monospace; font-size: 36px; font-weight: 700; text-align: center; padding: 20px 0; border-top: 1px solid #1e2030; border-bottom: 1px solid #1e2030; margin: 14px 0; }
.badge {
    display: inline-block; font-size: 9px; font-weight: 700;
    padding: 3px 10px; border-radius: 4px; letter-spacing: 0.8px;
}
.badge-MANDATORY { background: rgba(226,75,74,0.15); color: #f09595; border: 1px solid rgba(226,75,74,0.3); }
.badge-WARNING   { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-INFO      { background: rgba(34,197,94,0.1);  color: #86efac; border: 1px solid rgba(34,197,94,0.25); }

/* ── Progress bar ── */
.prog-wrap { margin: 12px 0; }
.prog-label { display: flex; justify-content: space-between; font-size: 11px; color: #6b7280; margin-bottom: 6px; }
.prog-bar-bg { height: 6px; background: #1e2030; border-radius: 3px; }
.prog-bar-fill { height: 6px; border-radius: 3px; transition: width 0.5s; }

/* ── Info boxes ── */
.meaning-box { background: rgba(226,75,74,0.06); border-left: 2px solid #e24b4a; border-radius: 0 8px 8px 0; padding: 10px 14px; margin-bottom: 10px; }
.action-box  { background: rgba(34,197,94,0.06); border-left: 2px solid #22c55e; border-radius: 0 8px 8px 0; padding: 10px 14px; }
.box-label { font-size: 9px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px; }
.box-text  { font-size: 13px; line-height: 1.6; }

/* ── Top 3 bars ── */
.bar-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.bar-name { font-size: 12px; color: #9ca3af; width: 155px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-bg   { flex: 1; height: 5px; background: #1e2030; border-radius: 3px; }
.bar-fill { height: 5px; border-radius: 3px; }
.bar-pct  { font-size: 11px; color: #6b7280; width: 38px; text-align: right; font-family: 'Geist Mono', monospace; }

/* ── No sign ── */
.no-sign {
    background: #13141c; border: 1px dashed #1e2030;
    border-radius: 10px; padding: 40px 20px;
    text-align: center; margin-bottom: 14px;
}
.no-sign-icon  { font-size: 42px; margin-bottom: 12px; }
.no-sign-title { font-size: 16px; font-weight: 600; color: #6b7280; margin-bottom: 6px; }
.no-sign-sub   { font-size: 12px; color: #4b5563; line-height: 1.6; }

/* ── Live webcam ── */
.live-detect-header {
    background: rgba(226,75,74,0.06); border: 1px solid rgba(226,75,74,0.2);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 12px;
}
.live-sign-name { font-size: 24px; font-weight: 700; color: #f0f0f0; }
.live-conf { font-family: 'Geist Mono', monospace; font-size: 48px; font-weight: 700; text-align: center; padding: 24px 0; }
.latency-row { display: flex; justify-content: space-between; font-size: 12px; color: #6b7280; margin-top: 10px; }

/* ── Info table ── */
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #1e2030; font-size: 12px; }
.info-row:last-child { border-bottom: none; }
.info-key { color: #6b7280; }
.info-val { color: #f0f0f0; font-weight: 500; font-family: 'Geist Mono', monospace; font-size: 11px; }

/* ── Pills ── */
.pills-wrap { display: flex; flex-wrap: wrap; gap: 6px; }
.pill { background: #1e2030; border: 1px solid #2a2d3e; border-radius: 20px; padding: 4px 12px; font-size: 11px; color: #9ca3af; display: inline-flex; align-items: center; gap: 6px; }
.pill-conf { font-family: 'Geist Mono', monospace; color: #e24b4a; }

/* ── Comparison table ── */
.comp-table { width: 100%; border-collapse: collapse; }
.comp-table th { font-size: 11px; font-weight: 600; color: #6b7280; letter-spacing: 0.5px; padding: 8px 12px; border-bottom: 1px solid #1e2030; text-align: left; }
.comp-table td { font-size: 12px; padding: 10px 12px; border-bottom: 1px solid #1e2030; }
.comp-table tr:last-child td { border-bottom: none; }
.comp-table .metric { color: #d1d5db; font-weight: 500; }
.comp-table .val { font-family: 'Geist Mono', monospace; color: #9ca3af; text-align: center; }

/* ── Section label ── */
.sec-label { font-size: 9px; font-weight: 700; color: #4b5563; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px; }

/* ── Process time ── */
.proc-time {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2);
    border-radius: 20px; padding: 4px 12px; font-size: 11px; color: #22c55e;
    font-family: 'Geist Mono', monospace; margin-bottom: 12px;
}

/* ── Streamlit overrides ── */
div[data-testid="stImage"] img { border-radius: 10px; width: 100%; }
.stFileUploader > div { background: transparent !important; border: none !important; }
div[data-testid="stCameraInput"] video { border-radius: 10px !important; }
div[data-testid="stCameraInput"] { background: #13141c !important; border: 1px solid #1e2030 !important; border-radius: 10px !important; }
.stButton > button {
    background: #e24b4a !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    padding: 8px 20px !important; width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button[kind="secondary"] { background: #1e2030 !important; color: #9ca3af !important; }
.stButton > button[kind="secondary"]:hover { color: #f0f0f0 !important; }
.stSpinner > div { border-top-color: #e24b4a !important; }
.stCheckbox label { color: #9ca3af !important; font-size: 13px !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .navbar { height: auto; flex-wrap: wrap; padding: 10px 14px; gap: 10px; }
    .nav-right { width: 100%; justify-content: space-between; }
    .model-switcher { flex: 1; }
    .model-btn { flex: 1; text-align: center; }
    .content { padding: 12px 14px; }
    .stat-value { font-size: 22px; }
    .pred-name { font-size: 17px; }
    .pred-conf-big { font-size: 44px; }
    .live-conf { font-size: 56px; }
    .live-sign-name { font-size: 20px; }
    .bar-name { width: 110px; font-size: 11px; }
    .tabs-bar { margin: 10px 14px 0; }
}
</style>
"""

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("tab", 0), ("mk", "gtsrb"), ("history", []), ("live_result", None)]:
    if k not in st.session_state: st.session_state[k] = v

models_dict = load_models()
all_labels  = load_all_labels()
summary     = load_summary()

mk           = st.session_state.mk
model, mok   = models_dict[mk]
labels       = all_labels[mk]
both_ok      = models_dict["indian"][1] and models_dict["gtsrb"][1]

st.markdown(CSS, unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🚦</div>
    <div>
      <div class="nav-brand">TrafficSense<span>AI</span></div>
      <div class="nav-sub">Sign Recognition · ADAS</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="model-switcher">
      <button class="model-btn {'active' if mk=='indian' else ''}" onclick="void(0)">🇮🇳 Indian</button>
      <button class="model-btn {'active' if mk=='gtsrb' else ''}" onclick="void(0)">🌍 GTSRB</button>
    </div>
    <div class="status-pill">
      <div class="status-dot"></div>
      {'Both Models Ready' if both_ok else 'Partial'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model + Tab switcher ────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    if st.button("🇮🇳 Indian", key="btn_indian", type="primary" if mk=="indian" else "secondary"):
        st.session_state.mk = "indian"; st.rerun()
with c2:
    if st.button("🌍 GTSRB", key="btn_gtsrb", type="primary" if mk=="gtsrb" else "secondary"):
        st.session_state.mk = "gtsrb"; st.rerun()
with c3:
    if st.button("📤 Upload", key="btn_upload", type="primary" if st.session_state.tab==0 else "secondary"):
        st.session_state.tab = 0; st.rerun()
with c4:
    if st.button("📷 Live", key="btn_live", type="primary" if st.session_state.tab==1 else "secondary"):
        st.session_state.tab = 1; st.rerun()
with c5:
    if st.button("📊 Stats", key="btn_stats", type="primary" if st.session_state.tab==2 else "secondary"):
        st.session_state.tab = 2; st.rerun()

st.markdown("<hr style='border-color:#1e2030;margin:0 0 4px;'>", unsafe_allow_html=True)

mk = st.session_state.mk
model, mok = models_dict[mk]
labels = all_labels[mk]
mk_label = "🇮🇳 Indian (59 classes)" if mk=="indian" else "🌍 GTSRB (43 classes)"

if not mok:
    st.error(f"❌ {mk_label} weights not found. Place the .pth file in the same folder as app.py.")

# helpers
def badge_html(danger):
    return f'<span class="badge badge-{danger}">{danger}</span>'

def render_pred(cid, conf, mk):
    name = labels.get(str(cid), f"Class {cid}")
    info = get_sign_info(mk, cid)
    color = info["color"]
    pct = conf*100
    st.markdown(f"""
    <div class="ts-card">
      <div class="ts-card-accent"></div>
      <div class="ts-card-body">
        <div class="pred-topbar">
          <div>
            <div class="pred-name">{name}</div>
            <div class="pred-sub">{mk_label} · Class #{cid}</div>
          </div>
          {badge_html(info['danger'])}
        </div>
        <div class="prog-wrap">
          <div class="prog-label"><span>Confidence</span><span style="font-family:'Geist Mono',monospace;font-weight:600;color:#f0f0f0;">{pct:.1f}%</span></div>
          <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div></div>
        </div>
        <div class="pred-conf-big" style="color:{color};">{pct:.1f}%</div>
        <div class="meaning-box">
          <div class="box-label" style="color:#e24b4a;">What this sign means</div>
          <div class="box-text" style="color:#d1d5db;">{info['meaning']}</div>
        </div>
        <div class="action-box">
          <div class="box-label" style="color:#22c55e;">ADAS Action</div>
          <div class="box-text" style="color:#86efac;">{info['action']}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_no_sign(conf):
    st.markdown(f"""
    <div class="no-sign">
      <div class="no-sign-icon">🔍</div>
      <div class="no-sign-title">No Traffic Sign Detected</div>
      <div class="no-sign-sub">Best confidence was only <strong style="color:#6b7280;">{conf*100:.1f}%</strong> — below the {NO_SIGN_THRESHOLD*100:.0f}% threshold.<br>Try a clearer image with the sign filling most of the frame.</div>
    </div>
    """, unsafe_allow_html=True)

def render_top3(ids, probs):
    colors = ["#e24b4a","#3b82f6","#7c3aed"]
    st.markdown('<div class="ts-card"><div class="ts-card-body"><div class="sec-label">Top 3 Predictions</div>', unsafe_allow_html=True)
    for idx, prob, color in zip(ids[:3], probs[:3], colors):
        name = labels.get(str(idx), f"Class {idx}")
        pct = prob*100
        st.markdown(f'<div class="bar-row"><span class="bar-name">{name}</span><div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div></div><span class="bar-pct">{pct:.1f}%</span></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.tab == 0:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    cl, cr = st.columns([1,1], gap="large")

    with cl:
        st.markdown(f'<div class="sec-label">Active Model — {mk_label}</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
            if mok:
                with st.spinner("Classifying…"):
                    ids, probs, ms = predict(model, img)
                st.markdown(f'<div class="proc-time">⚡ {ms:.0f} ms · {mk_label}</div>', unsafe_allow_html=True)
                if probs[0] < NO_SIGN_THRESHOLD:
                    render_no_sign(probs[0])
                else:
                    render_pred(ids[0], probs[0], mk)
                    render_top3(ids, probs)
                    st.session_state.history.insert(0,{"name":labels.get(str(ids[0]),f"Class {ids[0]}")[:22],"conf":probs[0]*100,"danger":get_sign_info(mk,ids[0])["danger"],"model":mk})
                    st.session_state.history = st.session_state.history[:8]
        else:
            st.markdown("""
            <div class="dropzone">
              <div class="dropzone-icon">🚦</div>
              <div class="dropzone-title">Drop your image here</div>
              <div class="dropzone-sub">or click Browse files above</div>
              <div class="dropzone-sub" style="margin-top:6px;">Supports PNG, JPG up to 200MB</div>
            </div>
            """, unsafe_allow_html=True)

    with cr:
        # Stats
        i_sum = summary.get(f"{mk}_model", summary)
        t1  = i_sum.get("final_val_acc",       76.81 if mk=="indian" else 83.42)
        t3  = i_sum.get("final_top3_acc",      92.80 if mk=="indian" else 97.09)
        n_det = len(st.session_state.history)
        avg_c = np.mean([h["conf"] for h in st.session_state.history]) if st.session_state.history else 0.0

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-label">Top-1 Val Acc</div><div class="stat-value" style="color:#22c55e;">{t1:.1f}%</div></div>
          <div class="stat-card"><div class="stat-label">Top-3 Val Acc</div><div class="stat-value" style="color:#3b82f6;">{t3:.1f}%</div></div>
          <div class="stat-card"><div class="stat-label">Signs Detected</div><div class="stat-value">{n_det}</div></div>
          <div class="stat-card"><div class="stat-label">Avg Confidence</div><div class="stat-value" style="color:#e24b4a;">{avg_c:.1f}%</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Model info
        n_cls = 59 if mk=="indian" else 43
        ds    = "Indian Signs (~13K)" if mk=="indian" else "GTSRB European (50K+)"
        rows  = [("Architecture","EfficientNet-B3"),("Dataset",ds),("Classes",str(n_cls)),("Backbone","ImageNet pretrained"),("Training","2-Phase Transfer Learning"),("Regulariser","Mixup + Label Smoothing")]
        st.markdown('<div class="sec-label" style="margin-top:4px;">Active Model Info</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ts-card"><div class="ts-card-body">', unsafe_allow_html=True)
        for k2,v2 in rows:
            st.markdown(f'<div class="info-row"><span class="info-key">{k2}</span><span class="info-val">{v2}</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Recent
        if st.session_state.history:
            st.markdown('<div class="sec-label" style="margin-top:4px;">Recent Predictions</div>', unsafe_allow_html=True)
            dc = {"MANDATORY":"#e24b4a","WARNING":"#f59e0b","INFO":"#22c55e"}
            pills = "".join(f'<span class="pill" style="border-left:2px solid {dc.get(h["danger"],"#22c55e")};">{"🇮🇳" if h["model"]=="indian" else "🌍"} {h["name"]} <span class="pill-conf">{h["conf"]:.0f}%</span></span>' for h in st.session_state.history)
            st.markdown(f'<div class="pills-wrap">{pills}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.tab == 1:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="live-detect-header">
      <span style="font-size:22px;">📡</span>
      <div>
        <div style="font-size:13px;font-weight:600;color:#93c5fd;">Live Autodetect — {mk_label}</div>
        <div style="font-size:11px;color:#6b7280;margin-top:2px;">Point camera at a traffic sign. Capture → auto-classify.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cl, cr = st.columns([1.1,1], gap="large")
    with cl:
        st.markdown('<div class="sec-label">Camera</div>', unsafe_allow_html=True)
        cam = st.camera_input("Point at a traffic sign", label_visibility="collapsed", key=f"cam_{mk}")
        auto = st.checkbox("🔄 Auto-refresh after each capture", value=True)

        if cam and mok:
            img = Image.open(cam).convert("RGB")
            with st.spinner("Detecting…"):
                ids, probs, ms = predict(model, img)
            no_sign = probs[0] < NO_SIGN_THRESHOLD
            st.session_state.live_result = {"ids":ids,"probs":probs,"ms":ms,"no_sign":no_sign,"cid":ids[0],"conf":probs[0]}
            if not no_sign:
                st.session_state.history.insert(0,{"name":labels.get(str(ids[0]),f"Class {ids[0]}")[:22],"conf":probs[0]*100,"danger":get_sign_info(mk,ids[0])["danger"],"model":mk})
                st.session_state.history = st.session_state.history[:8]

        r = st.session_state.live_result
        if r:
            st.markdown(f'<div class="proc-time">⚡ {r["ms"]:.0f} ms · {mk_label}</div>', unsafe_allow_html=True)
            if r["no_sign"]:
                st.markdown(f"""
                <div class="ts-card"><div class="ts-card-body" style="text-align:center;padding:32px 18px;">
                  <div style="font-size:40px;margin-bottom:10px;">🔍</div>
                  <div style="font-size:16px;font-weight:600;color:#6b7280;">No Traffic Sign Detected</div>
                  <div style="font-size:12px;color:#4b5563;margin-top:6px;">Confidence {r['conf']*100:.1f}% — below {NO_SIGN_THRESHOLD*100:.0f}% threshold.<br>Point camera directly at a sign.</div>
                </div></div>
                """, unsafe_allow_html=True)
            else:
                cid   = r["cid"]; conf = r["conf"]
                name  = labels.get(str(cid), f"Class {cid}")
                info  = get_sign_info(mk, cid)
                color = info["color"]
                st.markdown(f"""
                <div class="ts-card">
                  <div class="ts-card-accent"></div>
                  <div class="ts-card-body">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                      <div>
                        <div style="font-size:9px;color:#6b7280;letter-spacing:1px;margin-bottom:4px;">DETECTED SIGN</div>
                        <div class="live-sign-name">{name}</div>
                        {badge_html(info['danger'])}
                      </div>
                    </div>
                    <div class="live-conf" style="color:{color};">{conf*100:.1f}%</div>
                    <div class="action-box" style="margin-top:0;">
                      <div class="box-label" style="color:#22c55e;">ADAS Action</div>
                      <div class="box-text" style="color:#86efac;">{info['action']}</div>
                    </div>
                    <div class="latency-row">
                      <span>Model: {mk_label}</span>
                      <span>Latency: {r['ms']:.0f}ms</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        if auto and cam:
            time.sleep(0.4); st.rerun()

    with cr:
        if r and not r["no_sign"]:
            render_top3(r["ids"], r["probs"])

        if st.session_state.history:
            st.markdown('<div class="sec-label">Recent Detections</div>', unsafe_allow_html=True)
            dc = {"MANDATORY":"#e24b4a","WARNING":"#f59e0b","INFO":"#22c55e"}
            st.markdown('<div class="ts-card"><div class="ts-card-body">', unsafe_allow_html=True)
            for h in st.session_state.history[:6]:
                c_ = dc.get(h["danger"],"#22c55e")
                st.markdown(f'<div class="info-row"><span style="font-size:12px;color:#d1d5db;">{"🇮🇳" if h["model"]=="indian" else "🌍"} {h["name"]}</span><span style="font-size:12px;color:{c_};font-family:\'Geist Mono\',monospace;font-weight:600;">{h["conf"]:.1f}%</span></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        n_det = len(st.session_state.history)
        avg_c = np.mean([h["conf"] for h in st.session_state.history]) if st.session_state.history else 0
        st.markdown(f"""
        <div class="ts-card"><div class="ts-card-body">
          <div class="sec-label">Session Stats</div>
          <div style="font-family:'Geist Mono',monospace;font-size:40px;font-weight:700;color:#22c55e;">{n_det}</div>
          <div style="font-size:11px;color:#6b7280;margin-bottom:10px;">signs detected this session</div>
          <div style="height:4px;background:#1e2030;border-radius:2px;margin-bottom:10px;"><div style="width:{min(n_det*8,100)}%;height:4px;background:#22c55e;border-radius:2px;"></div></div>
          <div style="font-size:11px;color:#6b7280;">Avg confidence: <span style="color:#f0f0f0;font-family:'Geist Mono',monospace;">{avg_c:.1f}%</span></div>
        </div></div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", key="clear"):
            st.session_state.history = []; st.session_state.live_result = None; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL STATS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.tab == 2:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    cl, cr = st.columns([1,1], gap="large")

    i_s = summary.get("indian_model", {})
    g_s = summary.get("gtsrb_model", {})
    i_t1 = i_s.get("final_val_acc",76.81); i_t3 = i_s.get("final_top3_acc",92.80)
    i_p1 = i_s.get("phase1_best_val_acc",49.19); i_p2 = i_s.get("phase2_best_val_acc",76.81)
    g_t1 = g_s.get("final_val_acc",83.42); g_t3 = g_s.get("final_top3_acc",97.09)
    g_p1 = g_s.get("phase1_best_val_acc",39.53); g_p2 = g_s.get("phase2_best_val_acc",83.42)

    with cl:
        st.markdown('<div class="sec-label">Model Comparison</div>', unsafe_allow_html=True)
        comp = [
            ("Top-1 Val Accuracy", f"{i_t1:.1f}%", f"{g_t1:.1f}%"),
            ("Top-3 Val Accuracy", f"{i_t3:.1f}%", f"{g_t3:.1f}%"),
            ("Phase 1 Best",       f"{i_p1:.1f}%", f"{g_p1:.1f}%"),
            ("Phase 2 Best",       f"{i_p2:.1f}%", f"{g_p2:.1f}%"),
            ("Total Classes",      "59",            "43"),
            ("Training Images",    "~13K",          "50K+"),
            ("Architecture",       "EfficientNet-B3","EfficientNet-B3"),
            ("Training Strategy",  "2-Phase TL",    "2-Phase TL"),
        ]
        rows_html = "".join(f'<tr><td class="metric">{m}</td><td class="val" style="color:#93c5fd;">{iv}</td><td class="val" style="color:#fca5a5;">{gv}</td></tr>' for m,iv,gv in comp)
        st.markdown(f"""
        <div class="ts-card"><div class="ts-card-body">
          <table class="comp-table">
            <thead><tr><th>Metric</th><th style="text-align:center;">🇮🇳 Indian</th><th style="text-align:center;">🌍 GTSRB</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div></div>
        """, unsafe_allow_html=True)

        # Training curve
        st.markdown('<div class="sec-label" style="margin-top:4px;">Training Accuracy Curves</div>', unsafe_allow_html=True)
        ih1 = i_s.get("training_history",{}).get("phase1",{})
        ih2 = i_s.get("training_history",{}).get("phase2",{})
        gh1 = g_s.get("training_history",{}).get("phase1",{})
        gh2 = g_s.get("training_history",{}).get("phase2",{})
        i_val = ih1.get("val_acc",[]) + ih2.get("val_acc",[])
        g_val = gh1.get("val_acc",[]) + gh2.get("val_acc",[])
        if not i_val: i_val = [28,31,37,35,41,37,41,43,41,45,45,45,43,42,46,45,48,49,48,49,49,48,47,48,47,57,61,65,67,70,69,71,71,73,74,75,75,75,76,76,76,76,76]
        if not g_val: g_val = [19,25,30,28,34,30,34,34,33,36,34,36,37,37,37,38,38,38,38,38,38,38,39,39,39,49,54,58,61,65,68,71,73,75,77,79,80,81,81,82,81,82,83]

        fig, ax = plt.subplots(figsize=(7,3.2))
        fig.patch.set_facecolor("#13141c"); ax.set_facecolor("#0c0d12")
        p1i = len(ih1.get("val_acc", i_val[:25]))
        ax.axvspan(0.5, p1i+0.5, alpha=0.04, color="#3b82f6")
        ax.axvline(p1i+0.5, color="#1e2030", linestyle=":", lw=1)
        ax.plot(range(1,len(i_val)+1), i_val, color="#93c5fd", lw=2, label="🇮🇳 Indian", marker="o", markersize=2)
        ax.plot(range(1,len(g_val)+1), g_val, color="#fca5a5", lw=2, label="🌍 GTSRB",  marker="o", markersize=2)
        ax.text(p1i/2, min(min(i_val),min(g_val))-4, "Phase 1", ha="center", fontsize=8, color="#3b82f6", style="italic")
        ax.text(p1i+(len(i_val)-p1i)/2, min(min(i_val),min(g_val))-4, "Phase 2", ha="center", fontsize=8, color="#f59e0b", style="italic")
        ax.set_xlabel("Epoch", fontsize=9, color="#6b7280"); ax.set_ylabel("Val Accuracy (%)", fontsize=9, color="#6b7280")
        ax.tick_params(colors="#6b7280", labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor("#1e2030")
        ax.grid(True, color="#1e2030", alpha=0.6, lw=0.5)
        ax.legend(fontsize=9, labelcolor="#d1d5db", facecolor="#13141c", edgecolor="#1e2030")
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#13141c"); buf.seek(0); plt.close()
        st.image(buf, use_column_width=True)

    with cr:
        for label_txt, s, color, t1, t3, p1, p2, n_cls, ds in [
            ("🇮🇳 Indian Model", i_s, "#93c5fd", i_t1, i_t3, i_p1, i_p2, 59, "Indian Signs (~13K images)"),
            ("🌍 GTSRB Model",   g_s, "#fca5a5", g_t1, g_t3, g_p1, g_p2, 43, "GTSRB European (50K+ images)"),
        ]:
            detail = [("Architecture","EfficientNet-B3"),("Dataset",ds),("Classes",str(n_cls)),("Top-1 Accuracy",f"{t1:.2f}%"),("Top-3 Accuracy",f"{t3:.2f}%"),("Phase 1 Best",f"{p1:.2f}%"),("Phase 2 Best",f"{p2:.2f}%")]
            st.markdown(f'<div class="sec-label" style="margin-top:4px;">{label_txt}</div>', unsafe_allow_html=True)
            st.markdown('<div class="ts-card"><div class="ts-card-body">', unsafe_allow_html=True)
            for k2,v2 in detail:
                vc = color if "Accuracy" in k2 or "Best" in k2 else "#f0f0f0"
                st.markdown(f'<div class="info-row"><span class="info-key">{k2}</span><span class="info-val" style="color:{vc};">{v2}</span></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        shared = [("Framework","PyTorch + torchvision"),("Optimiser","AdamW"),("Scheduler","CosineAnnealingLR"),("Regularisation","Mixup + Label Smoothing"),("Dropout","0.4 / 0.3 / 0.2"),("Hardware","Kaggle P100 GPU")]
        st.markdown('<div class="sec-label" style="margin-top:4px;">Shared Config</div>', unsafe_allow_html=True)
        st.markdown('<div class="ts-card"><div class="ts-card-body">', unsafe_allow_html=True)
        for k2,v2 in shared:
            st.markdown(f'<div class="info-row"><span class="info-key">{k2}</span><span class="info-val">{v2}</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1e2030;margin-top:32px;padding:16px 24px;display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:12px;color:#4b5563;">TrafficSense AI — Advanced Driver Assistance System</span>
  <span style="font-size:11px;color:#4b5563;font-family:'Geist Mono',monospace;">EfficientNet-B3 · PyTorch · Streamlit</span>
</div>
""", unsafe_allow_html=True)