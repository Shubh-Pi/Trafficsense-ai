import os
import io
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
DEVICE   = torch.device("cpu")
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

# Confidence threshold below which we say "no sign detected"
NO_SIGN_THRESHOLD = 0.30

# ── Model Architecture (must match training) ──────────────────────────────────
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = efficientnet_b3(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base(x)


@st.cache_resource
def load_models():
    results = {}
    configs = {
        "indian": {"file": "indian_traffic_sign_model.pth", "classes": 59, "label_file": "indian_labels.json"},
        "gtsrb":  {"file": "gtsrb_traffic_sign_model.pth",  "classes": 43, "label_file": "gtsrb_labels.json"},
    }
    for key, cfg in configs.items():
        model = TrafficSignNet(num_classes=cfg["classes"])
        pth_candidates = [
            cfg["file"],
            f"/app/{cfg['file']}",
            os.path.join(os.path.dirname(__file__), cfg["file"]),
        ]
        loaded = False
        for path in pth_candidates:
            if not os.path.exists(path):
                continue
            try:
                obj   = torch.load(path, map_location=DEVICE, weights_only=False)
                state = obj if isinstance(obj, dict) else obj.state_dict()
                missing, _ = model.load_state_dict(state, strict=False)
                coverage   = 1 - len(missing) / len(model.state_dict())
                if coverage >= 0.85:
                    model.eval()
                    loaded = True
                    print(f"[TrafficSense] ✅ {key} model loaded ({coverage:.1%})")
                    break
            except Exception as e:
                print(f"[TrafficSense] {key} load error: {e}")
        results[key] = (model, loaded)
    return results


@st.cache_data
def load_all_labels():
    out = {}
    for key, fname in [("indian", "indian_labels.json"), ("gtsrb", "gtsrb_labels.json")]:
        paths = [fname, f"/app/{fname}", os.path.join(os.path.dirname(__file__), fname)]
        for p in paths:
            if os.path.exists(p):
                with open(p) as f:
                    out[key] = json.load(f)
                break
        if key not in out:
            out[key] = {str(i): f"Class {i}" for i in range(59 if key == "indian" else 43)}
    return out


@st.cache_data
def load_summary():
    for fname in ["training_summary.json", "/app/training_summary.json"]:
        if os.path.exists(fname):
            with open(fname) as f:
                return json.load(f)
    return {}


# ── GTSRB Sign info ───────────────────────────────────────────────────────────
GTSRB_INFO = {
    "0":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 20 km/h — very low speed zone.", "action": "Reduce to 20 km/h immediately."},
    "1":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 30 km/h — residential/school zone.", "action": "Reduce to 30 km/h. Watch for pedestrians."},
    "2":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 50 km/h — standard urban limit.", "action": "Maintain at or below 50 km/h."},
    "3":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 60 km/h — urban arterial road.", "action": "Reduce to 60 km/h."},
    "4":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 70 km/h — road leaving urban area.", "action": "Reduce to 70 km/h."},
    "5":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 80 km/h — rural road.", "action": "Reduce to 80 km/h."},
    "6":  {"danger": "INFO",      "color": "#1d9e75", "meaning": "End of 80 km/h restriction.", "action": "Restriction lifted. Drive to conditions."},
    "7":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 100 km/h — expressway.", "action": "Do not exceed 100 km/h."},
    "8":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Speed limit 120 km/h — motorway.", "action": "Do not exceed 120 km/h."},
    "9":  {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "No overtaking for all vehicles.", "action": "Stay in lane. Do not overtake."},
    "10": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "No overtaking for vehicles over 3.5t.", "action": "Heavy vehicles must not overtake."},
    "11": {"danger": "INFO",      "color": "#1d9e75", "meaning": "Right of way at next intersection.", "action": "You have priority. Maintain speed."},
    "12": {"danger": "INFO",      "color": "#1d9e75", "meaning": "Priority road — you have continuous right of way.", "action": "Maintain speed. You have priority."},
    "13": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Yield — give way to main road traffic.", "action": "Slow down and yield to all traffic."},
    "14": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Stop sign — complete stop required.", "action": "Stop completely before the line."},
    "15": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "No vehicles permitted on this road.", "action": "Do not enter. Find alternative route."},
    "16": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Vehicles over 3.5t prohibited.", "action": "Heavy vehicles must use alternative route."},
    "17": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "No entry — one-way road opposite direction.", "action": "Do not enter. Turn around."},
    "18": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "General hazard warning ahead.", "action": "Reduce speed. Be alert for hazards."},
    "19": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Dangerous left curve ahead.", "action": "Reduce speed before the curve."},
    "20": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Dangerous right curve ahead.", "action": "Reduce speed before the curve."},
    "21": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Double curve ahead.", "action": "Reduce speed significantly. Keep to lane."},
    "22": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Bumpy road surface ahead.", "action": "Reduce speed to protect suspension."},
    "23": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Slippery road ahead.", "action": "Reduce speed. Avoid sudden inputs."},
    "24": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Road narrows on the right.", "action": "Move centre-left. Watch oncoming traffic."},
    "25": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Roadworks ahead.", "action": "Reduce speed. Watch for workers."},
    "26": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Traffic signals ahead.", "action": "Prepare to stop at red light."},
    "27": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Pedestrian crossing ahead.", "action": "Slow down. Yield to pedestrians."},
    "28": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Children crossing — school zone.", "action": "Reduce speed sharply. Watch for children."},
    "29": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Bicycle crossing ahead.", "action": "Check mirrors. Give cyclists space."},
    "30": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Ice or snow likely on road.", "action": "Reduce speed. Increase following distance."},
    "31": {"danger": "WARNING",   "color": "#EF9F27", "meaning": "Wild animals crossing ahead.", "action": "Reduce speed. Scan road edges."},
    "32": {"danger": "INFO",      "color": "#1d9e75", "meaning": "End of all restrictions.", "action": "Normal rules apply. Drive to conditions."},
    "33": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Turn right ahead — mandatory.", "action": "Signal right. Prepare to turn right."},
    "34": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Turn left ahead — mandatory.", "action": "Signal left. Prepare to turn left."},
    "35": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Proceed straight only.", "action": "Do not turn. Go straight only."},
    "36": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Go straight or turn right only.", "action": "Left turns not permitted."},
    "37": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Go straight or turn left only.", "action": "Right turns not permitted."},
    "38": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Keep right.", "action": "Pass obstacle on the right side."},
    "39": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Keep left.", "action": "Pass obstacle on the left side."},
    "40": {"danger": "MANDATORY", "color": "#e24b4a", "meaning": "Roundabout — yield to circulating traffic.", "action": "Yield before entering roundabout."},
    "41": {"danger": "INFO",      "color": "#1d9e75", "meaning": "End of no-overtaking zone.", "action": "Overtaking now permitted where safe."},
    "42": {"danger": "INFO",      "color": "#1d9e75", "meaning": "End of no-overtaking zone for heavy vehicles.", "action": "Heavy vehicles may overtake where safe."},
}

INDIAN_INFO_DEFAULT = {"danger": "INFO", "color": "#378ADD", "meaning": "Indian road sign detected.", "action": "Follow standard road rules and local traffic laws."}


def get_sign_info(model_key, class_id):
    if model_key == "gtsrb":
        return GTSRB_INFO.get(str(class_id), {"danger": "INFO", "color": "#1d9e75", "meaning": "No description available.", "action": "Follow standard road rules."})
    return INDIAN_INFO_DEFAULT


# ── Transforms ────────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def flatten_alpha(img):
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        mask = img.split()[-1] if img.mode == "RGBA" else None
        bg.paste(img.convert("RGB"), mask=mask)
        return bg
    return img.convert("RGB")


def predict(model, img_pil):
    t0     = time.time()
    tensor = val_transform(flatten_alpha(img_pil)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    ms = (time.time() - t0) * 1000
    topk_probs, topk_ids = probs.topk(5)
    return topk_ids.cpu().numpy(), topk_probs.cpu().numpy(), ms


def is_no_sign(top_conf):
    return float(top_conf) < NO_SIGN_THRESHOLD


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #08090d !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Topbar ── */
.topbar {
    background: #0d0f18;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1e2235;
    position: sticky;
    top: 0;
    z-index: 100;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #e24b4a, #ff7043);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; box-shadow: 0 4px 14px rgba(226,75,74,0.35);
}
.brand-name { font-family: 'Syne', sans-serif; font-size: 17px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.3px; }
.brand-sub  { font-size: 10px; color: #475569; margin-top: 1px; letter-spacing: 0.3px; }
.model-switcher {
    display: flex; gap: 6px;
    background: #13161f;
    border: 1px solid #1e2235;
    border-radius: 10px;
    padding: 4px;
}
.model-btn {
    padding: 6px 14px;
    border-radius: 7px;
    font-size: 12px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    color: #64748b;
    background: transparent;
}
.model-btn.active-indian { background: #1a3a6b; color: #7eb8f7; }
.model-btn.active-gtsrb  { background: #3a1515; color: #f09595; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: #0a1f12; border: 1px solid #1d9e75;
    border-radius: 20px; padding: 5px 12px;
    font-size: 11px; color: #5DCAA5; font-weight: 500;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #1d9e75; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Tabs ── */
.tab-row {
    display: flex; gap: 4px;
    padding: 10px 24px 0;
    border-bottom: 1px solid #1e2235;
    background: #0d0f18;
}
.stab {
    padding: 10px 18px;
    font-size: 12px; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    color: #475569; background: none;
    border: none; border-bottom: 2px solid transparent;
    cursor: pointer; transition: all 0.2s;
    letter-spacing: 0.3px;
}
.stab:hover { color: #94a3b8; }
.stab.on { color: #e24b4a; border-bottom-color: #e24b4a; }

/* ── Content ── */
.content { padding: 20px 24px; }

/* ── Cards ── */
.card {
    background: #0d0f18;
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
}

/* ── Prediction card ── */
.pred-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 14px; }
.pred-name { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: #f1f5f9; line-height: 1.2; }
.pred-conf { font-family: 'DM Mono', monospace; font-size: 28px; font-weight: 500; }
.badge {
    display: inline-block; font-size: 9px; font-weight: 700;
    padding: 3px 9px; border-radius: 20px; letter-spacing: 0.8px;
    margin-top: 5px;
}
.badge-MANDATORY { background: #3a1515; color: #f09595; }
.badge-WARNING   { background: #2a2200; color: #FAC775; }
.badge-INFO      { background: #0a1f12; color: #9FE1CB; }

/* ── Info boxes ── */
.info-box {
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin-top: 10px;
}
.info-label { font-size: 9px; letter-spacing: 1px; margin-bottom: 4px; font-weight: 600; }
.info-text  { font-size: 13px; line-height: 1.6; }

/* ── Bar chart ── */
.bar-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.bar-name { font-size: 12px; color: #94a3b8; width: 160px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-bg   { flex: 1; height: 5px; background: #1e2235; border-radius: 3px; }
.bar-fill { height: 5px; border-radius: 3px; transition: width 0.5s ease; }
.bar-pct  { font-size: 11px; color: #475569; width: 40px; text-align: right; font-family: 'DM Mono', monospace; }

/* ── No sign card ── */
.no-sign-card {
    background: #0d0f18;
    border: 1px dashed #2a2d3a;
    border-radius: 14px;
    padding: 32px 20px;
    text-align: center;
    margin-bottom: 14px;
}
.no-sign-icon { font-size: 40px; margin-bottom: 12px; }
.no-sign-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: #64748b; margin-bottom: 6px; }
.no-sign-sub   { font-size: 12px; color: #475569; line-height: 1.6; }

/* ── Sec label ── */
.sec-label {
    font-size: 9px; font-weight: 700;
    color: #475569; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 10px;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed #1e2235;
    border-radius: 14px;
    padding: 36px 20px;
    text-align: center;
    background: #0a0c14;
    cursor: pointer;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #e24b4a44; }

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.stat-card { background: #0d0f18; border: 1px solid #1e2235; border-radius: 12px; padding: 14px; }
.stat-val  { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 700; }
.stat-lbl  { font-size: 10px; color: #475569; margin-top: 3px; letter-spacing: 0.3px; }

/* ── Info table ── */
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #1e2235; font-size: 12px; }
.info-row:last-child { border-bottom: none; }
.info-key { color: #475569; }
.info-val { color: #e2e8f0; font-weight: 500; font-family: 'DM Mono', monospace; font-size: 11px; }

/* ── History pills ── */
.h-pill { background: #0d0f18; border: 1px solid #1e2235; border-radius: 6px; padding: 4px 10px; font-size: 11px; color: #64748b; display: inline-block; margin: 3px; }

/* ── Process time ── */
.process-time {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0a1f12; border: 1px solid #1d9e75;
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; color: #5DCAA5; margin-bottom: 12px;
    font-family: 'DM Mono', monospace;
}

/* ── Live feed ── */
.live-result {
    background: #0d0f18;
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
}
.live-sign-name { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #f1f5f9; }
.live-conf { font-family: 'DM Mono', monospace; font-size: 32px; font-weight: 500; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .topbar { padding: 10px 14px; flex-wrap: wrap; gap: 8px; }
    .brand-name { font-size: 15px; }
    .model-switcher { order: 3; width: 100%; justify-content: center; }
    .content { padding: 12px 14px; }
    .tab-row { padding: 8px 14px 0; overflow-x: auto; }
    .pred-name { font-size: 17px; }
    .pred-conf { font-size: 24px; }
    .stat-grid { grid-template-columns: 1fr 1fr; }
    .live-sign-name { font-size: 18px; }
    .live-conf { font-size: 26px; }
    .bar-name { width: 110px; font-size: 11px; }
}

/* ── Streamlit widget overrides ── */
div[data-testid="stImage"] img { border-radius: 12px; width: 100%; }
.stFileUploader > div { background: transparent !important; border: none !important; }
div[data-testid="stCameraInput"] video { border-radius: 12px !important; }
div[data-testid="stCameraInput"] { background: #0d0f18 !important; border: 1px solid #1e2235 !important; border-radius: 14px !important; }
.stButton > button {
    background: linear-gradient(135deg, #e24b4a, #ff6b35) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.3px !important;
    padding: 10px 22px !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(226,75,74,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(226,75,74,0.5) !important; transform: translateY(-1px) !important; }
.stSpinner > div { border-top-color: #e24b4a !important; }
</style>
"""


# ── HTML helpers ───────────────────────────────────────────────────────────────
def badge(danger):
    return f'<span class="badge badge-{danger}">{danger}</span>'


def render_pred_card(class_id, conf, model_key, labels):
    name = labels.get(str(class_id), f"Class {class_id}")
    info = get_sign_info(model_key, class_id)
    color = info["color"]
    conf_pct = conf * 100

    st.markdown(f"""
    <div class="card">
      <div class="pred-header">
        <div>
          <div class="pred-name">{name}</div>
          {badge(info['danger'])}
        </div>
        <div class="pred-conf" style="color:{color};">{conf_pct:.1f}%</div>
      </div>
      <div style="height:5px;background:#1e2235;border-radius:3px;margin-bottom:14px;">
        <div style="width:{min(conf_pct,100):.1f}%;height:5px;background:{color};border-radius:3px;"></div>
      </div>
      <div class="info-box" style="background:#16101044;border-left:3px solid {color};">
        <div class="info-label" style="color:{color};">WHAT THIS SIGN MEANS</div>
        <div class="info-text" style="color:#cbd5e1;">{info['meaning']}</div>
      </div>
      <div class="info-box" style="background:#0a1f12;border-left:3px solid #1d9e75;margin-top:8px;">
        <div class="info-label" style="color:#1d9e75;">ADAS ACTION</div>
        <div class="info-text" style="color:#9FE1CB;">{info['action']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_no_sign_card(conf):
    st.markdown(f"""
    <div class="no-sign-card">
      <div class="no-sign-icon">🔍</div>
      <div class="no-sign-title">No Traffic Sign Detected</div>
      <div class="no-sign-sub">
        The model's best confidence was only <strong style="color:#64748b;">{conf*100:.1f}%</strong>
        — below the {NO_SIGN_THRESHOLD*100:.0f}% threshold.<br>
        Try a clearer image with the sign occupying most of the frame.
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_top3(ids, probs, labels):
    colors = ["#e24b4a", "#378ADD", "#7c5cbf"]
    st.markdown('<div class="card"><div class="sec-label">Top 3 Predictions</div>', unsafe_allow_html=True)
    for idx, prob, color in zip(ids[:3], probs[:3], colors):
        name = labels.get(str(idx), f"Class {idx}")
        pct  = prob * 100
        st.markdown(
            f'<div class="bar-row">'
            f'<span class="bar-name">{name}</span>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div></div>'
            f'<span class="bar-pct">{pct:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "active_tab"  not in st.session_state: st.session_state.active_tab  = 0
if "model_key"   not in st.session_state: st.session_state.model_key   = "gtsrb"
if "history"     not in st.session_state: st.session_state.history     = []
if "live_result" not in st.session_state: st.session_state.live_result = None

# ── Load resources ─────────────────────────────────────────────────────────────
models_dict = load_models()
all_labels  = load_all_labels()
summary     = load_summary()

mk          = st.session_state.model_key
model, model_ok = models_dict[mk]
labels      = all_labels[mk]

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

# ── Topbar ─────────────────────────────────────────────────────────────────────
indian_ok = models_dict["indian"][1]
gtsrb_ok  = models_dict["gtsrb"][1]
both_ok   = indian_ok and gtsrb_ok

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🚦</div>
    <div>
      <div class="brand-name">TrafficSense AI</div>
      <div class="brand-sub">Dual-Model Traffic Sign Recognition · ADAS</div>
    </div>
  </div>
  <div class="status-pill">
    <div class="status-dot"></div>
    {'Both Models Ready' if both_ok else 'Partial Load'}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model selector + Tabs ──────────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns([1, 1, 1, 1, 1])
with col_m1:
    if st.button("🇮🇳 Indian Model" if mk != "indian" else "🇮🇳 Indian Model ✓",
                 key="sel_indian",
                 type="primary" if mk == "indian" else "secondary"):
        st.session_state.model_key = "indian"
        st.rerun()
with col_m2:
    if st.button("🌍 GTSRB Model" if mk != "gtsrb" else "🌍 GTSRB Model ✓",
                 key="sel_gtsrb",
                 type="primary" if mk == "gtsrb" else "secondary"):
        st.session_state.model_key = "gtsrb"
        st.rerun()
with col_m3:
    if st.button("📤 Upload", key="tab_upload",
                 type="primary" if st.session_state.active_tab == 0 else "secondary"):
        st.session_state.active_tab = 0; st.rerun()
with col_m4:
    if st.button("📷 Live Webcam", key="tab_live",
                 type="primary" if st.session_state.active_tab == 1 else "secondary"):
        st.session_state.active_tab = 1; st.rerun()
with col_m5:
    if st.button("📊 Model Stats", key="tab_stats",
                 type="primary" if st.session_state.active_tab == 2 else "secondary"):
        st.session_state.active_tab = 2; st.rerun()

st.markdown("<hr style='border-color:#1e2235;margin:0;'>", unsafe_allow_html=True)

# model / labels reflect current selection
mk      = st.session_state.model_key
model, model_ok = models_dict[mk]
labels  = all_labels[mk]

model_label = "🇮🇳 Indian (58 classes)" if mk == "indian" else "🌍 GTSRB (43 classes)"

if not model_ok:
    st.error(f"❌ {model_label} weights not found. Place the .pth file in the same folder as app.py.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == 0:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown(f'<div class="sec-label">Active Model — {model_label}</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload traffic sign image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, use_column_width=True)

            if model_ok:
                with st.spinner("Analysing image…"):
                    ids, probs, ms = predict(model, img_pil)

                st.markdown(f'<div class="process-time">⚡ {ms:.0f} ms inference · {model_label}</div>', unsafe_allow_html=True)

                if is_no_sign(probs[0]):
                    render_no_sign_card(probs[0])
                else:
                    render_pred_card(ids[0], probs[0], mk, labels)
                    render_top3(ids, probs, labels)

                    # History
                    st.session_state.history.insert(0, {
                        "name":   labels.get(str(ids[0]), f"Class {ids[0]}")[:22],
                        "conf":   probs[0] * 100,
                        "danger": get_sign_info(mk, ids[0])["danger"],
                        "model":  mk,
                    })
                    st.session_state.history = st.session_state.history[:8]
        else:
            st.markdown("""
            <div class="upload-zone">
              <div style="font-size:36px;margin-bottom:10px;">🚦</div>
              <div style="color:#e24b4a;font-weight:600;font-size:14px;margin-bottom:6px;">Drop or browse a traffic sign image</div>
              <div style="color:#475569;font-size:12px;">PNG · JPG · JPEG</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        # Recent predictions
        if st.session_state.history:
            st.markdown('<div class="sec-label">Recent Predictions</div>', unsafe_allow_html=True)
            dc = {"MANDATORY": "#e24b4a", "WARNING": "#EF9F27", "INFO": "#1d9e75"}
            pills = "".join(
                f'<span class="h-pill" style="border-left:3px solid {dc.get(h["danger"],"#1d9e75")};">'
                f'{"🇮🇳" if h["model"]=="indian" else "🌍"} {h["name"]} · {h["conf"]:.0f}%</span>'
                for h in st.session_state.history
            )
            st.markdown(f'<div style="margin-bottom:16px;">{pills}</div>', unsafe_allow_html=True)

        # Model quick stats
        sum_key = mk  # "indian" or "gtsrb"
        m_sum   = summary.get(f"{sum_key}_model", summary)
        t1  = m_sum.get("final_val_acc",       76.81 if mk == "indian" else 83.42)
        t3  = m_sum.get("final_top3_acc",      92.80 if mk == "indian" else 97.09)
        p1b = m_sum.get("phase1_best_val_acc", 49.19 if mk == "indian" else 39.53)
        p2b = m_sum.get("phase2_best_val_acc", 76.81 if mk == "indian" else 83.42)

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-val" style="color:#1d9e75;">{t1:.1f}%</div>
            <div class="stat-lbl">Top-1 Val Accuracy</div>
          </div>
          <div class="stat-card">
            <div class="stat-val" style="color:#378ADD;">{t3:.1f}%</div>
            <div class="stat-lbl">Top-3 Val Accuracy</div>
          </div>
          <div class="stat-card">
            <div class="stat-val" style="color:#EF9F27;">{p1b:.1f}%</div>
            <div class="stat-lbl">Phase 1 Best</div>
          </div>
          <div class="stat-card">
            <div class="stat-val" style="color:#e24b4a;">{p2b:.1f}%</div>
            <div class="stat-lbl">Phase 2 Best</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Model info
        st.markdown('<div class="sec-label" style="margin-top:6px;">Model Info</div>', unsafe_allow_html=True)
        n_classes = 59 if mk == "indian" else 43
        dataset   = "Indian Traffic Signs (13K)" if mk == "indian" else "GTSRB European (50K+)"
        rows = [
            ("Architecture", "EfficientNet-B3"),
            ("Dataset",      dataset),
            ("Classes",      str(n_classes)),
            ("Backbone",     "ImageNet pretrained"),
            ("Training",     "2-Phase Transfer Learning"),
            ("Regulariser",  "Mixup + Label Smoothing"),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k, v in rows:
            st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM (continuous autodetect)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == 1:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0a1020;border:1px solid #1e2235;border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">
      <span style="font-size:20px;">📡</span>
      <div>
        <div style="font-size:13px;font-weight:600;color:#7eb8f7;">Live Autodetect Mode — {model_label}</div>
        <div style="font-size:11px;color:#475569;margin-top:2px;">Point your camera at a traffic sign. The app will continuously capture and classify every few seconds.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        st.markdown('<div class="sec-label">Camera Feed — Auto Capture</div>', unsafe_allow_html=True)

        # Continuous autodetect via repeated camera_input captures
        cam_img = st.camera_input(
            "Point at a traffic sign",
            label_visibility="collapsed",
            key=f"cam_{mk}"
        )

        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (keep re-capturing)", value=True)

        if cam_img and model_ok:
            img_pil = Image.open(cam_img).convert("RGB")
            with st.spinner("Detecting…"):
                ids, probs, ms = predict(model, img_pil)

            no_sign = is_no_sign(probs[0])
            st.session_state.live_result = {
                "ids": ids, "probs": probs, "ms": ms,
                "no_sign": no_sign,
                "class_id": ids[0], "conf": probs[0],
            }

            if not no_sign:
                st.session_state.history.insert(0, {
                    "name":   labels.get(str(ids[0]), f"Class {ids[0]}")[:22],
                    "conf":   probs[0] * 100,
                    "danger": get_sign_info(mk, ids[0])["danger"],
                    "model":  mk,
                })
                st.session_state.history = st.session_state.history[:8]

        # Show result below camera
        if st.session_state.live_result:
            r       = st.session_state.live_result
            ms      = r["ms"]
            no_sign = r["no_sign"]

            st.markdown(f'<div class="process-time">⚡ {ms:.0f} ms · {model_label}</div>', unsafe_allow_html=True)

            if no_sign:
                st.markdown(f"""
                <div class="live-result" style="border-color:#2a2d3a;text-align:center;padding:24px;">
                  <div style="font-size:36px;margin-bottom:8px;">🔍</div>
                  <div style="font-size:16px;font-weight:600;color:#64748b;">No Traffic Sign Detected</div>
                  <div style="font-size:11px;color:#475569;margin-top:6px;">Confidence {r['conf']*100:.1f}% — below {NO_SIGN_THRESHOLD*100:.0f}% threshold.<br>Point camera directly at a sign.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                cls_id = r["class_id"]
                conf   = r["conf"]
                name   = labels.get(str(cls_id), f"Class {cls_id}")
                info   = get_sign_info(mk, cls_id)
                color  = info["color"]

                st.markdown(f"""
                <div class="live-result">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                    <div>
                      <div style="font-size:9px;color:#475569;letter-spacing:1px;margin-bottom:4px;">DETECTED SIGN</div>
                      <div class="live-sign-name">{name}</div>
                      {badge(info['danger'])}
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:9px;color:#475569;letter-spacing:1px;margin-bottom:4px;">CONFIDENCE</div>
                      <div class="live-conf" style="color:{color};">{conf*100:.1f}%</div>
                    </div>
                  </div>
                  <div style="height:5px;background:#1e2235;border-radius:3px;margin-bottom:14px;">
                    <div style="width:{min(conf*100,100):.1f}%;height:5px;background:{color};border-radius:3px;"></div>
                  </div>
                  <div class="info-box" style="background:#0a1f12;border-left:3px solid #1d9e75;">
                    <div class="info-label" style="color:#1d9e75;">ADAS ACTION</div>
                    <div class="info-text" style="color:#9FE1CB;">{info['action']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        if auto_refresh and cam_img:
            time.sleep(0.5)
            st.rerun()

    with col_r:
        # Top 3
        if st.session_state.live_result and not st.session_state.live_result["no_sign"]:
            r = st.session_state.live_result
            render_top3(r["ids"], r["probs"], labels)

        # Recent detections
        if st.session_state.history:
            st.markdown('<div class="sec-label">Recent Detections</div>', unsafe_allow_html=True)
            dc = {"MANDATORY": "#e24b4a", "WARNING": "#EF9F27", "INFO": "#1d9e75"}
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for h in st.session_state.history[:6]:
                color = dc.get(h["danger"], "#1d9e75")
                st.markdown(
                    f'<div class="info-row">'
                    f'<span style="font-size:12px;color:#e2e8f0;">{"🇮🇳" if h["model"]=="indian" else "🌍"} {h["name"]}</span>'
                    f'<span style="font-size:12px;color:{color};font-weight:600;font-family:\'DM Mono\',monospace;">{h["conf"]:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # Session stats
        n_det    = len(st.session_state.history)
        avg_conf = np.mean([h["conf"] for h in st.session_state.history]) if st.session_state.history else 0
        st.markdown(f"""
        <div class="card" style="margin-top:4px;">
          <div class="sec-label">Session Stats</div>
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
            <span style="font-family:'Syne',sans-serif;font-size:32px;font-weight:700;color:#1d9e75;">{n_det}</span>
            <span style="font-size:11px;color:#475569;">signs detected</span>
          </div>
          <div style="height:4px;background:#1e2235;border-radius:2px;margin-bottom:10px;">
            <div style="width:{min(n_det*8,100)}%;height:4px;background:#1d9e75;border-radius:2px;"></div>
          </div>
          <div style="font-size:11px;color:#475569;">Avg confidence: <span style="color:#e2e8f0;font-family:'DM Mono',monospace;">{avg_conf:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", key="clear_hist"):
            st.session_state.history = []
            st.session_state.live_result = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL STATS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == 2:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1], gap="large")

    # Pull stats from summary JSON or use actuals
    indian_sum = summary.get("indian_model", {})
    gtsrb_sum  = summary.get("gtsrb_model",  {})

    i_t1  = indian_sum.get("final_val_acc",       76.81)
    i_t3  = indian_sum.get("final_top3_acc",      92.80)
    i_p1  = indian_sum.get("phase1_best_val_acc", 49.19)
    i_p2  = indian_sum.get("phase2_best_val_acc", 76.81)

    g_t1  = gtsrb_sum.get("final_val_acc",        83.42)
    g_t3  = gtsrb_sum.get("final_top3_acc",       97.09)
    g_p1  = gtsrb_sum.get("phase1_best_val_acc",  39.53)
    g_p2  = gtsrb_sum.get("phase2_best_val_acc",  83.42)

    with col_l:
        st.markdown('<div class="sec-label">Both Models — Performance Comparison</div>', unsafe_allow_html=True)

        # Comparison table
        comp_rows = [
            ("Top-1 Val Accuracy", f"{i_t1:.1f}%", f"{g_t1:.1f}%"),
            ("Top-3 Val Accuracy", f"{i_t3:.1f}%", f"{g_t3:.1f}%"),
            ("Phase 1 Best",       f"{i_p1:.1f}%", f"{g_p1:.1f}%"),
            ("Phase 2 Best",       f"{i_p2:.1f}%", f"{g_p2:.1f}%"),
            ("Classes",            "59",            "43"),
            ("Training Images",    "~13K",          "50K+"),
            ("Phase 1 Epochs",     str(indian_sum.get("phase1_epochs", 25)), str(gtsrb_sum.get("phase1_epochs", 25))),
            ("Phase 2 Epochs",     str(indian_sum.get("phase2_epochs", 20)), str(gtsrb_sum.get("phase2_epochs", 20))),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-row">'
            '<span class="info-key">Metric</span>'
            '<span style="font-size:11px;color:#7eb8f7;font-weight:600;">🇮🇳 Indian</span>'
            '<span style="font-size:11px;color:#f09595;font-weight:600;">🌍 GTSRB</span>'
            '</div>',
            unsafe_allow_html=True
        )
        for metric, iv, gv in comp_rows:
            st.markdown(
                f'<div class="info-row">'
                f'<span class="info-key">{metric}</span>'
                f'<span style="font-size:11px;color:#7eb8f7;font-family:\'DM Mono\',monospace;">{iv}</span>'
                f'<span style="font-size:11px;color:#f09595;font-family:\'DM Mono\',monospace;">{gv}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Training curves chart
        st.markdown('<div class="sec-label" style="margin-top:8px;">Training Curves (Phase 1 → Phase 2)</div>', unsafe_allow_html=True)

        ih1 = indian_sum.get("training_history", {}).get("phase1", {})
        ih2 = indian_sum.get("training_history", {}).get("phase2", {})
        gh1 = gtsrb_sum.get("training_history",  {}).get("phase1", {})
        gh2 = gtsrb_sum.get("training_history",  {}).get("phase2", {})

        i_val = ih1.get("val_acc", []) + ih2.get("val_acc", [])
        g_val = gh1.get("val_acc", []) + gh2.get("val_acc", [])

        # Fallback to actuals if no history saved
        if not i_val:
            i_val = [28,31,37,35,41,37,41,43,41,45,45,45,43,42,46,45,48,49,48,49,49,48,47,48,47,
                     57,61,65,67,70,69,71,71,73,74,75,75,75,76,76,76,76,75]
        if not g_val:
            g_val = [19,25,30,28,34,30,34,34,33,36,34,36,37,37,37,38,38,38,38,38,38,38,39,39,39,
                     49,54,58,61,65,68,71,73,75,77,79,80,81,81,82,81,82,83]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0d0f18")
        ax.set_facecolor("#0a0c14")

        ep_i = list(range(1, len(i_val) + 1))
        ep_g = list(range(1, len(g_val) + 1))
        p1_len_i = len(ih1.get("val_acc", i_val[:25]))
        p1_len_g = len(gh1.get("val_acc", g_val[:25]))

        ax.axvspan(0.5, p1_len_i + 0.5, alpha=0.05, color="#378ADD")
        ax.axvline(p1_len_i + 0.5, color="#1e2235", linestyle=":", lw=1)

        ax.plot(ep_i, i_val, color="#7eb8f7", lw=2, label="🇮🇳 Indian", marker="o", markersize=2)
        ax.plot(ep_g, g_val, color="#f09595", lw=2, label="🌍 GTSRB",  marker="o", markersize=2)

        ax.text(p1_len_i / 2, min(min(i_val), min(g_val)) - 3, "Phase 1", ha="center", fontsize=8, color="#378ADD", style="italic")
        ax.text(p1_len_i + (len(i_val) - p1_len_i) / 2, min(min(i_val), min(g_val)) - 3, "Phase 2", ha="center", fontsize=8, color="#EF9F27", style="italic")

        ax.set_xlabel("Epoch", fontsize=9, color="#475569")
        ax.set_ylabel("Val Accuracy (%)", fontsize=9, color="#475569")
        ax.tick_params(colors="#475569", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#1e2235")
        ax.grid(True, color="#1e2235", alpha=0.6, lw=0.5)
        ax.legend(fontsize=9, labelcolor="#e2e8f0", facecolor="#0d0f18", edgecolor="#1e2235")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d0f18")
        buf.seek(0); plt.close()
        st.image(buf, use_column_width=True)

    with col_r:
        # Indian model details
        st.markdown('<div class="sec-label">🇮🇳 Indian Model Details</div>', unsafe_allow_html=True)
        indian_details = [
            ("Architecture",  "EfficientNet-B3"),
            ("Dataset",       "Indian Signs ~13K images"),
            ("Classes",       "59"),
            ("Top-1 Accuracy",f"{i_t1:.2f}%"),
            ("Top-3 Accuracy",f"{i_t3:.2f}%"),
            ("Phase 1 Best",  f"{i_p1:.2f}%"),
            ("Phase 2 Best",  f"{i_p2:.2f}%"),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k, v in indian_details:
            color = "#7eb8f7" if "Accuracy" in k or "Best" in k else "#e2e8f0"
            st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val" style="color:{color};">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # GTSRB model details
        st.markdown('<div class="sec-label" style="margin-top:14px;">🌍 GTSRB Model Details</div>', unsafe_allow_html=True)
        gtsrb_details = [
            ("Architecture",  "EfficientNet-B3"),
            ("Dataset",       "GTSRB 50K+ images"),
            ("Classes",       "43"),
            ("Top-1 Accuracy",f"{g_t1:.2f}%"),
            ("Top-3 Accuracy",f"{g_t3:.2f}%"),
            ("Phase 1 Best",  f"{g_p1:.2f}%"),
            ("Phase 2 Best",  f"{g_p2:.2f}%"),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k, v in gtsrb_details:
            color = "#f09595" if "Accuracy" in k or "Best" in k else "#e2e8f0"
            st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val" style="color:{color};">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Shared details
        st.markdown('<div class="sec-label" style="margin-top:14px;">Shared Training Config</div>', unsafe_allow_html=True)
        shared = [
            ("Framework",        "PyTorch + torchvision"),
            ("Backbone",         "EfficientNet-B3 (ImageNet)"),
            ("Optimiser",        "AdamW"),
            ("Scheduler",        "CosineAnnealingLR"),
            ("Regularisation",   "Mixup + Label Smoothing"),
            ("Anti-overfitting", "Dropout 0.4/0.3/0.2"),
            ("Hardware",         "Kaggle P100 GPU"),
            ("Strategy",         "2-Phase Transfer Learning"),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k, v in shared:
            st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
