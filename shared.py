"""
shared.py — All shared logic for TrafficSense AI
Models, preprocessing, prediction, helpers, CSS, constants
"""
import os, io, json, time, csv, datetime
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3
import streamlit as st

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224
DEVICE = torch.device("cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
NO_SIGN_THRESHOLD = 0.25
LOW_CONF  = 0.60
HIGH_CONF = 0.95
MAX_HISTORY = 200
DANGER_ICONS  = {"MANDATORY": "🔴", "WARNING": "🟡", "INFO": "🟢"}
DANGER_COLORS = {"MANDATORY": "#e24b4a", "WARNING": "#f59e0b", "INFO": "#22c55e"}

# ── Model ──────────────────────────────────────────────────────────────────────
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = efficientnet_b3(weights=None)
        inf = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(inf, 512), nn.SiLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.SiLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.base(x)

@st.cache_resource
def load_models():
    out = {}
    for key, fname, n in [("indian","indian_traffic_sign_model.pth",59),("gtsrb","gtsrb_traffic_sign_model.pth",43)]:
        m = TrafficSignNet(n); loaded = False
        for p in [fname, f"/app/{fname}", os.path.join(os.path.dirname(__file__), fname)]:
            if not os.path.exists(p): continue
            try:
                obj   = torch.load(p, map_location=DEVICE, weights_only=False)
                state = obj if isinstance(obj, dict) else obj.state_dict()
                miss, _ = m.load_state_dict(state, strict=False)
                if 1 - len(miss) / len(m.state_dict()) >= 0.85:
                    m.eval(); loaded = True; break
            except: pass
        out[key] = (m, loaded)
    return out

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

# ── Sign metadata ──────────────────────────────────────────────────────────────
GTSRB_INFO = {
    "0":  {"danger":"MANDATORY","meaning":"Speed limit 20 km/h.","action":"Reduce to 20 km/h immediately."},
    "1":  {"danger":"MANDATORY","meaning":"Speed limit 30 km/h — residential zone.","action":"Reduce to 30 km/h. Watch for pedestrians."},
    "2":  {"danger":"MANDATORY","meaning":"Speed limit 50 km/h — standard urban limit.","action":"Maintain at or below 50 km/h."},
    "3":  {"danger":"MANDATORY","meaning":"Speed limit 60 km/h.","action":"Reduce to 60 km/h."},
    "4":  {"danger":"MANDATORY","meaning":"Speed limit 70 km/h.","action":"Reduce to 70 km/h."},
    "5":  {"danger":"MANDATORY","meaning":"Speed limit 80 km/h.","action":"Reduce to 80 km/h."},
    "6":  {"danger":"INFO","meaning":"End of 80 km/h restriction.","action":"Drive to conditions."},
    "7":  {"danger":"MANDATORY","meaning":"Speed limit 100 km/h.","action":"Do not exceed 100 km/h."},
    "8":  {"danger":"MANDATORY","meaning":"Speed limit 120 km/h.","action":"Do not exceed 120 km/h."},
    "9":  {"danger":"MANDATORY","meaning":"No overtaking for all vehicles.","action":"Stay in lane. Do not overtake."},
    "10": {"danger":"MANDATORY","meaning":"No overtaking for vehicles over 3.5t.","action":"Heavy vehicles must not overtake."},
    "11": {"danger":"INFO","meaning":"Right of way at next intersection.","action":"You have priority. Maintain speed."},
    "12": {"danger":"INFO","meaning":"Priority road — continuous right of way.","action":"Maintain speed. You have priority."},
    "13": {"danger":"WARNING","meaning":"Yield — give way to main road traffic.","action":"Slow down and yield to all traffic."},
    "14": {"danger":"MANDATORY","meaning":"Stop sign — complete stop required.","action":"Stop completely before the line."},
    "15": {"danger":"MANDATORY","meaning":"No vehicles permitted.","action":"Do not enter. Find alternative route."},
    "16": {"danger":"MANDATORY","meaning":"Vehicles over 3.5t prohibited.","action":"Heavy vehicles must use alternative route."},
    "17": {"danger":"MANDATORY","meaning":"No entry — one-way road.","action":"Do not enter. Turn around."},
    "18": {"danger":"WARNING","meaning":"General hazard warning ahead.","action":"Reduce speed. Be alert."},
    "19": {"danger":"WARNING","meaning":"Dangerous left curve ahead.","action":"Reduce speed before the curve."},
    "20": {"danger":"WARNING","meaning":"Dangerous right curve ahead.","action":"Reduce speed before the curve."},
    "21": {"danger":"WARNING","meaning":"Double curve ahead.","action":"Reduce speed. Keep to lane."},
    "22": {"danger":"WARNING","meaning":"Bumpy road ahead.","action":"Reduce speed to protect suspension."},
    "23": {"danger":"WARNING","meaning":"Slippery road ahead.","action":"Reduce speed. Avoid sudden inputs."},
    "24": {"danger":"WARNING","meaning":"Road narrows on the right.","action":"Move centre-left. Watch oncoming traffic."},
    "25": {"danger":"WARNING","meaning":"Roadworks ahead.","action":"Reduce speed. Watch for workers."},
    "26": {"danger":"WARNING","meaning":"Traffic signals ahead.","action":"Prepare to stop at red light."},
    "27": {"danger":"WARNING","meaning":"Pedestrian crossing ahead.","action":"Slow down. Yield to pedestrians."},
    "28": {"danger":"WARNING","meaning":"Children crossing — school zone.","action":"Reduce speed sharply. Watch for children."},
    "29": {"danger":"WARNING","meaning":"Bicycle crossing ahead.","action":"Check mirrors. Give cyclists space."},
    "30": {"danger":"WARNING","meaning":"Ice or snow likely.","action":"Reduce speed. Increase following distance."},
    "31": {"danger":"WARNING","meaning":"Wild animals crossing ahead.","action":"Reduce speed. Scan road edges."},
    "32": {"danger":"INFO","meaning":"End of all restrictions.","action":"Normal rules apply. Drive to conditions."},
    "33": {"danger":"MANDATORY","meaning":"Turn right — mandatory.","action":"Signal right. Prepare to turn right."},
    "34": {"danger":"MANDATORY","meaning":"Turn left — mandatory.","action":"Signal left. Prepare to turn left."},
    "35": {"danger":"MANDATORY","meaning":"Proceed straight only.","action":"Do not turn. Go straight only."},
    "36": {"danger":"MANDATORY","meaning":"Go straight or turn right only.","action":"Left turns not permitted."},
    "37": {"danger":"MANDATORY","meaning":"Go straight or turn left only.","action":"Right turns not permitted."},
    "38": {"danger":"MANDATORY","meaning":"Keep right.","action":"Pass obstacle on the right side."},
    "39": {"danger":"MANDATORY","meaning":"Keep left.","action":"Pass obstacle on the left side."},
    "40": {"danger":"MANDATORY","meaning":"Roundabout — yield to circulating traffic.","action":"Yield before entering roundabout."},
    "41": {"danger":"INFO","meaning":"End of no-overtaking zone.","action":"Overtaking now permitted where safe."},
    "42": {"danger":"INFO","meaning":"End of no-overtaking zone for heavy vehicles.","action":"Heavy vehicles may overtake where safe."},
}
INDIAN_DEFAULT = {"danger":"INFO","meaning":"Indian road sign detected.","action":"Follow standard road rules and local traffic laws."}

def get_info(mk, cid):
    if mk == "gtsrb": return GTSRB_INFO.get(str(cid), {"danger":"INFO","meaning":"No description.","action":"Follow standard rules."})
    return INDIAN_DEFAULT

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(img_pil):
    warns = []; changed = False
    try:
        from PIL import ImageOps
        img_pil = ImageOps.exif_transpose(img_pil)
    except: pass
    if img_pil.mode == "P" and "transparency" in img_pil.info:
        img_pil = img_pil.convert("RGBA")
    if img_pil.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img_pil.size, (255,255,255))
        mask = img_pil.split()[-1] if "A" in img_pil.mode else None
        bg.paste(img_pil.convert("RGB"), mask=mask); img_pil = bg
    elif img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    arr = np.array(img_pil, dtype=np.float32)
    if arr.shape[2] == 3:
        ch_std = np.std([arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()])
        if ch_std < 5: warns.append(("⚠️","Grayscale image — colour images yield better accuracy."))
    w, h = img_pil.size
    if w < 64 or h < 64:
        warns.append(("⚠️", f"Very small image ({w}×{h}px) — upscaling with Lanczos."))
        scale = max(128/w, 128/h)
        img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        arr = np.array(img_pil, dtype=np.float32); changed = True
    gray = np.mean(arr, axis=2).astype(np.float32)
    gy, gx = np.gradient(gray); lap_var = float(np.var(np.sqrt(gx**2 + gy**2)))
    if lap_var < 80: warns.append(("⚠️", f"Blurry image (sharpness:{lap_var:.0f}) — results may be less accurate."))
    brightness = arr.mean()
    if brightness < 50:
        warns.append(("🌙", "Dark image — applying CLAHE contrast enhancement."))
        if HAS_CV2:
            lab = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            arr = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB).astype(np.float32)
        else:
            arr = np.clip(arr * (120 / max(brightness, 1)), 0, 255)
        changed = True
    elif brightness > 210:
        warns.append(("☀️", "Overexposed image — applying gamma correction."))
        arr = np.clip(np.power(arr/255.0, 1.4)*255.0, 0, 255); changed = True
    if HAS_CV2:
        arr_u8 = arr.astype(np.uint8)
        arr_u8 = cv2.bilateralFilter(arr_u8, d=9, sigmaColor=75, sigmaSpace=75)
        noise_est = float(np.std(arr - arr_u8.astype(np.float32)))
        if noise_est > 12: arr_u8 = cv2.fastNlMeansDenoisingColored(arr_u8, None, 10,10,7,21)
        arr = arr_u8.astype(np.float32); changed = True
        lab = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        arr = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB).astype(np.float32)
    else:
        img_tmp = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
        arr = np.array(img_tmp, dtype=np.float32)
    img_s  = Image.fromarray(arr.astype(np.uint8))
    arr_s  = np.array(img_s, dtype=np.float32)
    arr_b  = np.array(img_s.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    arr    = np.clip(arr_s + 0.4*(arr_s - arr_b), 0, 255)
    img_final = Image.fromarray(arr.astype(np.uint8))
    W, H = img_final.size; side = min(W, H)
    img_final = img_final.crop(((W-side)//2, (H-side)//2, (W-side)//2+side, (H-side)//2+side))
    img_final = img_final.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    # Always ensure solid RGB output (no transparency channel that causes checkerboard display)
    if img_final.mode != "RGB":
        bg = Image.new("RGB", img_final.size, (255, 255, 255))
        if img_final.mode in ("RGBA", "LA"):
            bg.paste(img_final.convert("RGB"), mask=img_final.split()[-1])
        else:
            bg.paste(img_final.convert("RGB"))
        img_final = bg
    return img_final, warns, changed

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def predict(model, img_pil):
    t0 = time.time()
    tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    ms = (time.time() - t0) * 1000
    topk_probs, topk_ids = probs.topk(5)
    return topk_ids.cpu().numpy(), topk_probs.cpu().numpy(), ms

def make_csv(rows, cols):
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(cols)
    for r in rows: w.writerow(r)
    return buf.getvalue().encode()

# ── Render helpers ─────────────────────────────────────────────────────────────
def badge(danger):
    icon = DANGER_ICONS.get(danger, "🟢")
    cls  = {"MANDATORY":"bM","WARNING":"bW","INFO":"bI"}.get(danger, "bI")
    return f'<span class="badge {cls}">{icon} {danger}</span>'

def conf_ind(c):
    if c >= HIGH_CONF:   return '<div class="ci-h">✅ High Confidence — reliable prediction</div>'
    if NO_SIGN_THRESHOLD <= c < LOW_CONF: return '<div class="ci-l">⚠️ Low confidence — try better lighting or crop tighter</div>'
    return ""

def render_warns(ws):
    for icon, msg in ws:
        st.markdown(f'<div class="wbanner"><span>{icon}</span><span>{msg}</span></div>', unsafe_allow_html=True)

def render_pred(cid, conf, mk, mk_label, labels):
    name = labels.get(str(cid), f"Class {cid}"); info = get_info(mk, cid)
    color = DANGER_COLORS.get(info["danger"], "#22c55e"); pct = conf * 100
    st.markdown(f"""
<div class="card"><div class="ct"></div><div class="card-body">
  <div class="pred-hdr">
    <div><div class="pred-name">{name}</div><div class="pred-sub">{mk_label} · Class #{cid}</div></div>
    {badge(info['danger'])}
  </div>
  {conf_ind(conf)}
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#4b5563;margin-bottom:6px;">
    <span>Confidence</span>
    <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#f0f0f0;">{pct:.1f}%</span>
  </div>
  <div class="pb-bg"><div class="pb-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div></div>
  <div class="conf-big" style="color:{color};">{pct:.1f}%</div>
  <div class="mbox"><div class="blabel" style="color:#e24b4a;">What this sign means</div><div class="btext" style="color:#d1d5db;">{info['meaning']}</div></div>
  <div class="abox"><div class="blabel" style="color:#22c55e;">ADAS Action</div><div class="btext" style="color:#86efac;">{info['action']}</div></div>
</div></div>""", unsafe_allow_html=True)

def render_no_sign(conf):
    st.markdown(f"""
<div class="nosign">
  <div style="font-size:38px;margin-bottom:12px;">🔍</div>
  <div style="font-size:16px;font-weight:600;color:#4b5563;margin-bottom:6px;">No Traffic Sign Detected</div>
  <div style="font-size:12px;color:#374151;">Best confidence: <strong style="color:#4b5563;">{conf*100:.1f}%</strong> — below {NO_SIGN_THRESHOLD*100:.0f}% threshold</div>
  <div class="nosign-tips">
    <div style="font-size:10px;font-weight:700;color:#374151;letter-spacing:1px;margin-bottom:8px;">💡 TIPS TO IMPROVE DETECTION</div>
    <div class="ntip">📐 Crop tighter — sign should fill at least 60% of frame</div>
    <div class="ntip">💡 Ensure good lighting — avoid shadows or harsh backlight</div>
    <div class="ntip">📷 Hold steady — motion blur significantly reduces accuracy</div>
    <div class="ntip">🔄 Try the other model — Indian vs GTSRB sign sets differ</div>
    <div class="ntip">🖼 Use a clear front-facing angle — avoid extreme perspective</div>
  </div>
</div>""", unsafe_allow_html=True)

def render_top3(ids, probs, labels, mk):
    above20  = sum(1 for p in probs[:3] if p > 0.20)
    dangers  = [get_info(mk, int(ids[i]))["danger"] for i in range(3) if probs[i] > 0.20]
    multi    = above20 >= 3 and len(set(dangers)) >= 2
    colors   = ["#e24b4a", "#3b82f6", "#7c3aed"]
    st.markdown('<div class="card"><div class="card-body"><div class="slabel">Top 3 Predictions</div>', unsafe_allow_html=True)
    for idx, prob, color in zip(ids[:3], probs[:3], colors):
        name = labels.get(str(idx), f"Class {idx}"); pct = prob * 100
        icon = DANGER_ICONS.get(get_info(mk, int(idx))["danger"], "🟢")
        st.markdown(f'<div class="br"><span class="bn">{icon} {name}</span><div class="bb"><div class="bf" style="width:{min(pct,100):.1f}%;background:{color};"></div></div><span class="bp">{pct:.1f}%</span></div>', unsafe_allow_html=True)
    if multi:
        st.markdown('<div class="mwarn">⚠️ Multiple sign types detected — more than one sign may be in frame</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

def draw_sparkline(vals):
    if len(vals) < 2: return None
    fig, ax = plt.subplots(figsize=(4, 0.9))
    fig.patch.set_facecolor("#080a10"); ax.set_facecolor("#080a10")
    x = list(range(len(vals)))
    ax.plot(x, vals, color="#e24b4a", lw=1.5, marker="o", markersize=3)
    ax.fill_between(x, vals, alpha=0.1, color="#e24b4a")
    ax.set_xlim(0, max(len(vals)-1, 1)); ax.set_ylim(0, 1); ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#080a10")
    buf.seek(0); plt.close(); return buf

def draw_gauge(pct, color, label):
    fig, ax = plt.subplots(figsize=(2.6, 1.5), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#0d0f18")
    theta_val = np.pi - (np.pi * pct / 100)
    thetas = np.linspace(np.pi, 0, 100)
    ax.plot(thetas, [1]*100, color="#1a1d2e", lw=12, solid_capstyle="round")
    thetas2 = np.linspace(np.pi, theta_val, 100)
    ax.plot(thetas2, [1]*100, color=color, lw=12, solid_capstyle="round")
    ax.text(0, 0, f"{pct:.1f}%", ha="center", va="center", fontsize=13, fontweight="700", color=color, fontfamily="monospace")
    ax.text(0, -0.55, label, ha="center", va="center", fontsize=7, color="#4b5563")
    ax.set_ylim(0, 1.3); ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d0f18")
    buf.seek(0); plt.close(); return buf

# ── Shared CSS (base styles used by both desktop and mobile) ───────────────────
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:#080a10!important;color:#f0f0f0!important;font-family:'Inter',sans-serif!important;
}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}

/* Navbar */
.navbar{background:#0d0f18;border-bottom:1px solid #1a1d2e;padding:0 28px;display:flex;align-items:center;justify-content:space-between;height:58px;position:sticky;top:0;z-index:100;}
.nav-logo{display:flex;align-items:center;gap:12px;}
.nav-logo-icon{width:36px;height:36px;background:linear-gradient(135deg,rgba(226,75,74,0.2),rgba(226,75,74,0.05));border:1px solid rgba(226,75,74,0.35);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px;}
.nav-brand{font-size:16px;font-weight:700;color:#f0f0f0;letter-spacing:-0.3px;}
.nav-brand span{color:#e24b4a;}
.nav-sub{font-size:10px;color:#4b5563;margin-top:1px;}
.nav-right{display:flex;align-items:center;gap:10px;}
.model-sw{display:flex;background:#13151f;border:1px solid #1a1d2e;border-radius:9px;padding:3px;gap:2px;}
.model-btn{padding:5px 16px;border-radius:7px;font-size:12px;font-weight:600;border:none;cursor:pointer;transition:all 0.15s;color:#4b5563;background:transparent;font-family:'Inter',sans-serif;}
.model-btn.active{background:#e24b4a;color:#fff;box-shadow:0 2px 8px rgba(226,75,74,0.3);}
.status-pill{display:flex;align-items:center;gap:6px;background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:20px;padding:5px 14px;font-size:11px;color:#22c55e;font-weight:500;}
.sdot{width:6px;height:6px;border-radius:50%;background:#22c55e;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.25}}

/* Layout cards */
.wrap{padding:22px 28px;}
.card{background:#0d0f18;border:1px solid #1a1d2e;border-radius:12px;overflow:hidden;margin-bottom:14px;}
.card-body{padding:16px 20px;}
.ct{height:2px;background:linear-gradient(90deg,#e24b4a,rgba(226,75,74,0.1),transparent);}
.ctb{height:2px;background:linear-gradient(90deg,#3b82f6,rgba(59,130,246,0.1),transparent);}
.ctg{height:2px;background:linear-gradient(90deg,#22c55e,rgba(34,197,94,0.1),transparent);}

/* Upload dropzone */
.dz{border:2px dashed #1a1d2e;border-radius:12px;padding:52px 20px;text-align:center;background:#080a10;cursor:pointer;margin-bottom:14px;transition:border-color 0.2s;}
.dz:hover{border-color:rgba(226,75,74,0.3);}
.dz-icon{font-size:38px;margin-bottom:14px;}
.dz-title{font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:5px;}
.dz-sub{font-size:12px;color:#4b5563;}
.dz-fmt{font-size:11px;color:#374151;margin-top:8px;font-family:'JetBrains Mono',monospace;}

/* Stats grid */
.sg{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.sc{background:#0d0f18;border:1px solid #1a1d2e;border-radius:11px;padding:14px 16px;}
.sl{font-size:9px;font-weight:700;color:#374151;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:7px;}
.sv{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:#f0f0f0;}

/* Badges */
.badge{display:inline-flex;align-items:center;gap:4px;font-size:9px;font-weight:700;padding:3px 9px;border-radius:4px;letter-spacing:0.8px;}
.bM{background:rgba(226,75,74,0.12);color:#fca5a5;border:1px solid rgba(226,75,74,0.25);}
.bW{background:rgba(245,158,11,0.12);color:#fcd34d;border:1px solid rgba(245,158,11,0.25);}
.bI{background:rgba(34,197,94,0.08);color:#86efac;border:1px solid rgba(34,197,94,0.2);}

/* Confidence indicators */
.ci-h{display:inline-flex;align-items:center;gap:5px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:6px;padding:4px 10px;font-size:11px;color:#22c55e;margin-bottom:10px;}
.ci-l{display:inline-flex;align-items:center;gap:5px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:6px;padding:4px 10px;font-size:11px;color:#f59e0b;margin-bottom:10px;}
.wbanner{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:9px;padding:10px 14px;margin-bottom:10px;font-size:12px;color:#fcd34d;display:flex;align-items:flex-start;gap:8px;}

/* Prediction card */
.pred-hdr{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px;}
.pred-name{font-size:20px;font-weight:700;color:#f1f5f9;line-height:1.2;}
.pred-sub{font-size:11px;color:#4b5563;margin-top:3px;}
.conf-big{font-family:'JetBrains Mono',monospace;font-size:38px;font-weight:700;text-align:center;padding:20px 0;border-top:1px solid #1a1d2e;border-bottom:1px solid #1a1d2e;margin:14px 0;}
.pb-bg{height:6px;background:#1a1d2e;border-radius:3px;margin-bottom:14px;}
.pb-fill{height:6px;border-radius:3px;}
.mbox{background:rgba(226,75,74,0.05);border-left:2px solid #e24b4a;border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px;}
.abox{background:rgba(34,197,94,0.05);border-left:2px solid #22c55e;border-radius:0 8px 8px 0;padding:10px 14px;}
.blabel{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;}
.btext{font-size:13px;line-height:1.6;}

/* No-sign box */
.nosign{background:#0d0f18;border:1px dashed #1a1d2e;border-radius:12px;padding:38px 20px;text-align:center;margin-bottom:14px;}
.nosign-tips{background:#080a10;border:1px solid #1a1d2e;border-radius:8px;padding:12px 14px;margin-top:14px;text-align:left;}
.ntip{font-size:12px;color:#374151;padding:3px 0;display:flex;align-items:center;gap:8px;}

/* Top-3 bars */
.br{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.bn{font-size:12px;color:#6b7280;width:155px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.bb{flex:1;height:5px;background:#1a1d2e;border-radius:3px;}
.bf{height:5px;border-radius:3px;}
.bp{font-size:11px;color:#4b5563;width:38px;text-align:right;font-family:'JetBrains Mono',monospace;}
.mwarn{background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.2);border-radius:8px;padding:8px 12px;margin-top:8px;font-size:11px;color:#c4b5fd;}

/* Live tab */
.live-name{font-size:22px;font-weight:700;color:#f1f5f9;line-height:1.2;}
.live-conf{font-family:'JetBrains Mono',monospace;font-size:52px;font-weight:700;text-align:center;padding:22px 0;}
.latrow{display:flex;justify-content:space-between;font-size:11px;color:#4b5563;margin-top:10px;}

/* Info rows */
.ir{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #1a1d2e;font-size:12px;}
.ir:last-child{border-bottom:none;}
.ik{color:#4b5563;}
.iv{color:#f0f0f0;font-weight:500;font-family:'JetBrains Mono',monospace;font-size:11px;}

/* Tables */
.comptbl{width:100%;border-collapse:collapse;}
.comptbl th{font-size:10px;font-weight:700;color:#374151;letter-spacing:0.8px;padding:8px 12px;border-bottom:1px solid #1a1d2e;text-align:left;}
.comptbl td{font-size:12px;padding:10px 12px;border-bottom:1px solid #1a1d2e;}
.comptbl tr:last-child td{border-bottom:none;}
.met{color:#d1d5db;font-weight:500;}
.vv{font-family:'JetBrains Mono',monospace;text-align:center;}
.htbl{width:100%;border-collapse:collapse;}
.htbl th{font-size:10px;font-weight:700;color:#374151;letter-spacing:0.8px;padding:8px 12px;border-bottom:1px solid #1a1d2e;text-align:left;}
.htbl td{font-size:12px;padding:9px 12px;border-bottom:1px solid #1a1d2e;color:#9ca3af;}
.htbl tr:last-child td{border-bottom:none;}
.hname{color:#e2e8f0;font-weight:500;}
.hconf{font-family:'JetBrains Mono',monospace;font-weight:600;}

/* History most-detected card */
.most-det{background:linear-gradient(135deg,rgba(226,75,74,0.08),rgba(226,75,74,0.02));border:1px solid rgba(226,75,74,0.15);border-radius:12px;padding:16px 20px;margin-bottom:14px;display:flex;align-items:center;gap:16px;}
.most-det-icon{font-size:30px;}
.most-det-label{font-size:9px;color:#4b5563;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;}
.most-det-name{font-size:16px;font-weight:700;color:#f1f5f9;}
.most-det-count{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;margin-left:auto;}

/* Per-class bars */
.cab{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.cab-name{font-size:11px;color:#9ca3af;width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.cab-bg{flex:1;height:16px;background:#1a1d2e;border-radius:3px;overflow:hidden;}
.cab-fill{height:16px;border-radius:3px;display:flex;align-items:center;padding-left:6px;}
.cab-pct{font-size:9px;color:#fff;font-family:'JetBrains Mono',monospace;font-weight:700;}

/* Misc */
.slabel{font-size:9px;font-weight:700;color:#374151;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;}
.ptime{display:inline-flex;align-items:center;gap:6px;background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.18);border-radius:20px;padding:4px 12px;font-size:11px;color:#22c55e;font-family:'JetBrains Mono',monospace;margin-bottom:12px;}
.ba-label{font-size:10px;color:#4b5563;text-align:center;margin-bottom:4px;font-weight:600;letter-spacing:0.5px;}

/* Streamlit widget overrides */
div[data-testid="stImage"] img{border-radius:10px;width:100%;}
.stFileUploader>div{background:transparent!important;border:none!important;}
div[data-testid="stCameraInput"] video{border-radius:12px!important;}
div[data-testid="stCameraInput"]{background:#0d0f18!important;border:1px solid #1a1d2e!important;border-radius:12px!important;}
.stButton>button{background:#e24b4a!important;color:#fff!important;border:none!important;border-radius:9px!important;font-family:'Inter',sans-serif!important;font-weight:600!important;padding:9px 20px!important;width:100%!important;transition:opacity 0.15s!important;}
.stButton>button:hover{opacity:0.82!important;}
.stButton>button[kind="secondary"]{background:#13151f!important;color:#6b7280!important;border:1px solid #1a1d2e!important;}
.stButton>button[kind="secondary"]:hover{color:#f0f0f0!important;}
.stSpinner>div{border-top-color:#e24b4a!important;}
.stCheckbox label{color:#6b7280!important;font-size:12px!important;}
div[data-testid="stDownloadButton"]>button{background:#13151f!important;color:#22c55e!important;border:1px solid rgba(34,197,94,0.2)!important;font-size:12px!important;}
</style>
"""