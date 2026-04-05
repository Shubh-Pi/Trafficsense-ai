import os, io, json, time, csv, datetime
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

st.set_page_config(page_title="TrafficSense AI", page_icon="🚦", layout="wide", initial_sidebar_state="collapsed")

IMG_SIZE=224; DEVICE=torch.device("cpu")
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
NO_SIGN_THRESHOLD=0.30; LOW_CONF=0.60; HIGH_CONF=0.95
MAX_HISTORY=200
DANGER_ICONS={"MANDATORY":"🔴","WARNING":"🟡","INFO":"🟢"}
DANGER_COLORS={"MANDATORY":"#e24b4a","WARNING":"#f59e0b","INFO":"#22c55e"}

class TrafficSignNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.base=efficientnet_b3(weights=None)
        inf=self.base.classifier[1].in_features
        self.base.classifier=nn.Sequential(nn.Dropout(0.4),nn.Linear(inf,512),nn.SiLU(),nn.BatchNorm1d(512),nn.Dropout(0.3),nn.Linear(512,256),nn.SiLU(),nn.BatchNorm1d(256),nn.Dropout(0.2),nn.Linear(256,num_classes))
    def forward(self,x): return self.base(x)

@st.cache_resource
def load_models():
    out={}
    for key,fname,n in [("indian","indian_traffic_sign_model.pth",59),("gtsrb","gtsrb_traffic_sign_model.pth",43)]:
        m=TrafficSignNet(n); loaded=False
        for p in [fname,f"/app/{fname}",os.path.join(os.path.dirname(__file__),fname)]:
            if not os.path.exists(p): continue
            try:
                obj=torch.load(p,map_location=DEVICE,weights_only=False)
                state=obj if isinstance(obj,dict) else obj.state_dict()
                miss,_=m.load_state_dict(state,strict=False)
                if 1-len(miss)/len(m.state_dict())>=0.85: m.eval();loaded=True;break
            except: pass
        out[key]=(m,loaded)
    return out

@st.cache_data
def load_all_labels():
    out={}
    for key,fname,n in [("indian","indian_labels.json",59),("gtsrb","gtsrb_labels.json",43)]:
        for p in [fname,f"/app/{fname}",os.path.join(os.path.dirname(__file__),fname)]:
            if os.path.exists(p):
                with open(p) as f: out[key]=json.load(f);break
        if key not in out: out[key]={str(i):f"Class {i}" for i in range(n)}
    return out

@st.cache_data
def load_summary():
    for f in ["training_summary.json","/app/training_summary.json"]:
        if os.path.exists(f):
            with open(f) as fh: return json.load(fh)
    return {}

GTSRB_INFO={
    "0":{"danger":"MANDATORY","meaning":"Speed limit 20 km/h.","action":"Reduce to 20 km/h immediately."},
    "1":{"danger":"MANDATORY","meaning":"Speed limit 30 km/h — residential zone.","action":"Reduce to 30 km/h. Watch for pedestrians."},
    "2":{"danger":"MANDATORY","meaning":"Speed limit 50 km/h — standard urban limit.","action":"Maintain at or below 50 km/h."},
    "3":{"danger":"MANDATORY","meaning":"Speed limit 60 km/h.","action":"Reduce to 60 km/h."},
    "4":{"danger":"MANDATORY","meaning":"Speed limit 70 km/h.","action":"Reduce to 70 km/h."},
    "5":{"danger":"MANDATORY","meaning":"Speed limit 80 km/h.","action":"Reduce to 80 km/h."},
    "6":{"danger":"INFO","meaning":"End of 80 km/h restriction.","action":"Drive to conditions."},
    "7":{"danger":"MANDATORY","meaning":"Speed limit 100 km/h.","action":"Do not exceed 100 km/h."},
    "8":{"danger":"MANDATORY","meaning":"Speed limit 120 km/h.","action":"Do not exceed 120 km/h."},
    "9":{"danger":"MANDATORY","meaning":"No overtaking for all vehicles.","action":"Stay in lane. Do not overtake."},
    "10":{"danger":"MANDATORY","meaning":"No overtaking for vehicles over 3.5t.","action":"Heavy vehicles must not overtake."},
    "11":{"danger":"INFO","meaning":"Right of way at next intersection.","action":"You have priority. Maintain speed."},
    "12":{"danger":"INFO","meaning":"Priority road — continuous right of way.","action":"Maintain speed. You have priority."},
    "13":{"danger":"WARNING","meaning":"Yield — give way to main road traffic.","action":"Slow down and yield to all traffic."},
    "14":{"danger":"MANDATORY","meaning":"Stop sign — complete stop required.","action":"Stop completely before the line."},
    "15":{"danger":"MANDATORY","meaning":"No vehicles permitted.","action":"Do not enter. Find alternative route."},
    "16":{"danger":"MANDATORY","meaning":"Vehicles over 3.5t prohibited.","action":"Heavy vehicles must use alternative route."},
    "17":{"danger":"MANDATORY","meaning":"No entry — one-way road.","action":"Do not enter. Turn around."},
    "18":{"danger":"WARNING","meaning":"General hazard warning ahead.","action":"Reduce speed. Be alert."},
    "19":{"danger":"WARNING","meaning":"Dangerous left curve ahead.","action":"Reduce speed before the curve."},
    "20":{"danger":"WARNING","meaning":"Dangerous right curve ahead.","action":"Reduce speed before the curve."},
    "21":{"danger":"WARNING","meaning":"Double curve ahead.","action":"Reduce speed. Keep to lane."},
    "22":{"danger":"WARNING","meaning":"Bumpy road ahead.","action":"Reduce speed to protect suspension."},
    "23":{"danger":"WARNING","meaning":"Slippery road ahead.","action":"Reduce speed. Avoid sudden inputs."},
    "24":{"danger":"WARNING","meaning":"Road narrows on the right.","action":"Move centre-left. Watch oncoming traffic."},
    "25":{"danger":"WARNING","meaning":"Roadworks ahead.","action":"Reduce speed. Watch for workers."},
    "26":{"danger":"WARNING","meaning":"Traffic signals ahead.","action":"Prepare to stop at red light."},
    "27":{"danger":"WARNING","meaning":"Pedestrian crossing ahead.","action":"Slow down. Yield to pedestrians."},
    "28":{"danger":"WARNING","meaning":"Children crossing — school zone.","action":"Reduce speed sharply. Watch for children."},
    "29":{"danger":"WARNING","meaning":"Bicycle crossing ahead.","action":"Check mirrors. Give cyclists space."},
    "30":{"danger":"WARNING","meaning":"Ice or snow likely.","action":"Reduce speed. Increase following distance."},
    "31":{"danger":"WARNING","meaning":"Wild animals crossing ahead.","action":"Reduce speed. Scan road edges."},
    "32":{"danger":"INFO","meaning":"End of all restrictions.","action":"Normal rules apply. Drive to conditions."},
    "33":{"danger":"MANDATORY","meaning":"Turn right — mandatory.","action":"Signal right. Prepare to turn right."},
    "34":{"danger":"MANDATORY","meaning":"Turn left — mandatory.","action":"Signal left. Prepare to turn left."},
    "35":{"danger":"MANDATORY","meaning":"Proceed straight only.","action":"Do not turn. Go straight only."},
    "36":{"danger":"MANDATORY","meaning":"Go straight or turn right only.","action":"Left turns not permitted."},
    "37":{"danger":"MANDATORY","meaning":"Go straight or turn left only.","action":"Right turns not permitted."},
    "38":{"danger":"MANDATORY","meaning":"Keep right.","action":"Pass obstacle on the right side."},
    "39":{"danger":"MANDATORY","meaning":"Keep left.","action":"Pass obstacle on the left side."},
    "40":{"danger":"MANDATORY","meaning":"Roundabout — yield to circulating traffic.","action":"Yield before entering roundabout."},
    "41":{"danger":"INFO","meaning":"End of no-overtaking zone.","action":"Overtaking now permitted where safe."},
    "42":{"danger":"INFO","meaning":"End of no-overtaking zone for heavy vehicles.","action":"Heavy vehicles may overtake where safe."},
}
INDIAN_DEFAULT={"danger":"INFO","meaning":"Indian road sign detected.","action":"Follow standard road rules and local traffic laws."}

def get_info(mk,cid):
    if mk=="gtsrb": return GTSRB_INFO.get(str(cid),{"danger":"INFO","meaning":"No description.","action":"Follow standard rules."})
    return INDIAN_DEFAULT

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(img_pil):
    warns=[]; changed=False
    try:
        from PIL import ImageOps
        img_pil=ImageOps.exif_transpose(img_pil)
    except: pass
    if img_pil.mode=="P" and "transparency" in img_pil.info: img_pil=img_pil.convert("RGBA")
    if img_pil.mode in("RGBA","LA"):
        bg=Image.new("RGB",img_pil.size,(255,255,255))
        mask=img_pil.split()[-1] if "A" in img_pil.mode else None
        bg.paste(img_pil.convert("RGB"),mask=mask); img_pil=bg
    elif img_pil.mode!="RGB": img_pil=img_pil.convert("RGB")
    arr=np.array(img_pil,dtype=np.float32)
    if arr.shape[2]==3:
        ch_std=np.std([arr[:,:,0].mean(),arr[:,:,1].mean(),arr[:,:,2].mean()])
        if ch_std<5: warns.append(("⚠️","Grayscale image — colour images yield better accuracy."))
    w,h=img_pil.size
    if w<64 or h<64:
        warns.append(("⚠️",f"Very small image ({w}×{h}px) — upscaling with Lanczos."))
        scale=max(128/w,128/h); img_pil=img_pil.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
        arr=np.array(img_pil,dtype=np.float32); changed=True
    gray=np.mean(arr,axis=2).astype(np.float32)
    gy,gx=np.gradient(gray); lap_var=float(np.var(np.sqrt(gx**2+gy**2)))
    if lap_var<80: warns.append(("⚠️",f"Blurry image (sharpness:{lap_var:.0f}) — results may be less accurate."))
    brightness=arr.mean()
    if brightness<50:
        warns.append(("🌙","Dark image — applying CLAHE contrast enhancement."))
        if HAS_CV2:
            lab=cv2.cvtColor(arr.astype(np.uint8),cv2.COLOR_RGB2LAB)
            l,a,b=cv2.split(lab); clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
            l=clahe.apply(l); arr=cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2RGB).astype(np.float32)
        else: arr=np.clip(arr*(120/max(brightness,1)),0,255)
        changed=True
    elif brightness>210:
        warns.append(("☀️","Overexposed image — applying gamma correction."))
        arr=np.clip(np.power(arr/255.0,1.8)*255.0,0,255); changed=True
    if HAS_CV2:
        arr_u8=arr.astype(np.uint8)
        arr_u8=cv2.bilateralFilter(arr_u8,d=9,sigmaColor=75,sigmaSpace=75)
        noise_est=float(np.std(arr-arr_u8.astype(np.float32)))
        if noise_est>12: arr_u8=cv2.fastNlMeansDenoisingColored(arr_u8,None,10,10,7,21)
        arr=arr_u8.astype(np.float32); changed=True
        lab=cv2.cvtColor(arr.astype(np.uint8),cv2.COLOR_RGB2LAB)
        l,a,b=cv2.split(lab); clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l=clahe.apply(l); arr=cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2RGB).astype(np.float32)
    else:
        img_tmp=Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
        arr=np.array(img_tmp,dtype=np.float32)
    img_s=Image.fromarray(arr.astype(np.uint8))
    arr_s=np.array(img_s,dtype=np.float32)
    arr_b=np.array(img_s.filter(ImageFilter.GaussianBlur(1.0)),dtype=np.float32)
    arr=np.clip(arr_s+0.4*(arr_s-arr_b),0,255)
    img_final=Image.fromarray(arr.astype(np.uint8))
    W,H=img_final.size; side=min(W,H)
    img_final=img_final.crop(((W-side)//2,(H-side)//2,(W-side)//2+side,(H-side)//2+side))
    img_final=img_final.resize((IMG_SIZE,IMG_SIZE),Image.LANCZOS)
    return img_final,warns,changed

val_transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

def predict(model,img_pil):
    t0=time.time()
    tensor=val_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): probs=torch.softmax(model(tensor),dim=1)[0]
    ms=(time.time()-t0)*1000
    topk_probs,topk_ids=probs.topk(5)
    return topk_ids.cpu().numpy(),topk_probs.cpu().numpy(),ms

def make_csv(rows,cols):
    buf=io.StringIO(); w=csv.writer(buf); w.writerow(cols)
    for r in rows: w.writerow(r)
    return buf.getvalue().encode()

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS="""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#080a10!important;color:#f0f0f0!important;font-family:'Inter',sans-serif!important;}
[data-testid="stHeader"]{background:transparent!important;}[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
.navbar{background:#0d0f18;border-bottom:1px solid #1a1d2e;padding:0 28px;display:flex;align-items:center;justify-content:space-between;height:58px;position:sticky;top:0;z-index:100;}
.nav-logo{display:flex;align-items:center;gap:12px;}
.nav-logo-icon{width:36px;height:36px;background:linear-gradient(135deg,rgba(226,75,74,0.2),rgba(226,75,74,0.05));border:1px solid rgba(226,75,74,0.35);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px;}
.nav-brand{font-size:16px;font-weight:700;color:#f0f0f0;letter-spacing:-0.3px;}.nav-brand span{color:#e24b4a;}
.nav-sub{font-size:10px;color:#4b5563;margin-top:1px;}
.nav-right{display:flex;align-items:center;gap:10px;}
.model-sw{display:flex;background:#13151f;border:1px solid #1a1d2e;border-radius:9px;padding:3px;gap:2px;}
.model-btn{padding:5px 16px;border-radius:7px;font-size:12px;font-weight:600;border:none;cursor:pointer;transition:all 0.15s;color:#4b5563;background:transparent;font-family:'Inter',sans-serif;}
.model-btn.active{background:#e24b4a;color:#fff;box-shadow:0 2px 8px rgba(226,75,74,0.3);}
.status-pill{display:flex;align-items:center;gap:6px;background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:20px;padding:5px 14px;font-size:11px;color:#22c55e;font-weight:500;}
.sdot{width:6px;height:6px;border-radius:50%;background:#22c55e;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.25}}
.wrap{padding:22px 28px;}
.card{background:#0d0f18;border:1px solid #1a1d2e;border-radius:12px;overflow:hidden;margin-bottom:14px;}
.card-body{padding:16px 20px;}
.ct{height:2px;background:linear-gradient(90deg,#e24b4a,rgba(226,75,74,0.1),transparent);}
.ctb{height:2px;background:linear-gradient(90deg,#3b82f6,rgba(59,130,246,0.1),transparent);}
.ctg{height:2px;background:linear-gradient(90deg,#22c55e,rgba(34,197,94,0.1),transparent);}
.dz{border:2px dashed #1a1d2e;border-radius:12px;padding:52px 20px;text-align:center;background:#080a10;cursor:pointer;margin-bottom:14px;transition:border-color 0.2s;}
.dz:hover{border-color:rgba(226,75,74,0.3);}
.dz-icon{font-size:38px;margin-bottom:14px;}.dz-title{font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:5px;}
.dz-sub{font-size:12px;color:#4b5563;}.dz-fmt{font-size:11px;color:#374151;margin-top:8px;font-family:'JetBrains Mono',monospace;}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;}
.sc{background:#0d0f18;border:1px solid #1a1d2e;border-radius:11px;padding:14px 16px;}
.sl{font-size:9px;font-weight:700;color:#374151;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:7px;}
.sv{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:#f0f0f0;}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:9px;font-weight:700;padding:3px 9px;border-radius:4px;letter-spacing:0.8px;}
.bM{background:rgba(226,75,74,0.12);color:#fca5a5;border:1px solid rgba(226,75,74,0.25);}
.bW{background:rgba(245,158,11,0.12);color:#fcd34d;border:1px solid rgba(245,158,11,0.25);}
.bI{background:rgba(34,197,94,0.08);color:#86efac;border:1px solid rgba(34,197,94,0.2);}
.ci-h{display:inline-flex;align-items:center;gap:5px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:6px;padding:4px 10px;font-size:11px;color:#22c55e;margin-bottom:10px;}
.ci-l{display:inline-flex;align-items:center;gap:5px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:6px;padding:4px 10px;font-size:11px;color:#f59e0b;margin-bottom:10px;}
.wbanner{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:9px;padding:10px 14px;margin-bottom:10px;font-size:12px;color:#fcd34d;display:flex;align-items:flex-start;gap:8px;}
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
.nosign{background:#0d0f18;border:1px dashed #1a1d2e;border-radius:12px;padding:38px 20px;text-align:center;margin-bottom:14px;}
.nosign-tips{background:#080a10;border:1px solid #1a1d2e;border-radius:8px;padding:12px 14px;margin-top:14px;text-align:left;}
.ntip{font-size:12px;color:#374151;padding:3px 0;display:flex;align-items:center;gap:8px;}
.br{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.bn{font-size:12px;color:#6b7280;width:155px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.bb{flex:1;height:5px;background:#1a1d2e;border-radius:3px;}
.bf{height:5px;border-radius:3px;}
.bp{font-size:11px;color:#4b5563;width:38px;text-align:right;font-family:'JetBrains Mono',monospace;}
.mwarn{background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.2);border-radius:8px;padding:8px 12px;margin-top:8px;font-size:11px;color:#c4b5fd;}
.live-name{font-size:22px;font-weight:700;color:#f1f5f9;line-height:1.2;}
.live-conf{font-family:'JetBrains Mono',monospace;font-size:52px;font-weight:700;text-align:center;padding:22px 0;}
.latrow{display:flex;justify-content:space-between;font-size:11px;color:#4b5563;margin-top:10px;}
.ir{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #1a1d2e;font-size:12px;}
.ir:last-child{border-bottom:none;}.ik{color:#4b5563;}.iv{color:#f0f0f0;font-weight:500;font-family:'JetBrains Mono',monospace;font-size:11px;}
.comptbl{width:100%;border-collapse:collapse;}
.comptbl th{font-size:10px;font-weight:700;color:#374151;letter-spacing:0.8px;padding:8px 12px;border-bottom:1px solid #1a1d2e;text-align:left;}
.comptbl td{font-size:12px;padding:10px 12px;border-bottom:1px solid #1a1d2e;}
.comptbl tr:last-child td{border-bottom:none;}.met{color:#d1d5db;font-weight:500;}.vv{font-family:'JetBrains Mono',monospace;text-align:center;}
.htbl{width:100%;border-collapse:collapse;}
.htbl th{font-size:10px;font-weight:700;color:#374151;letter-spacing:0.8px;padding:8px 12px;border-bottom:1px solid #1a1d2e;text-align:left;}
.htbl td{font-size:12px;padding:9px 12px;border-bottom:1px solid #1a1d2e;color:#9ca3af;}
.htbl tr:last-child td{border-bottom:none;}.hname{color:#e2e8f0;font-weight:500;}.hconf{font-family:'JetBrains Mono',monospace;font-weight:600;}
.most-det{background:linear-gradient(135deg,rgba(226,75,74,0.08),rgba(226,75,74,0.02));border:1px solid rgba(226,75,74,0.15);border-radius:12px;padding:16px 20px;margin-bottom:14px;display:flex;align-items:center;gap:16px;}
.most-det-icon{font-size:30px;}.most-det-label{font-size:9px;color:#4b5563;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;}
.most-det-name{font-size:16px;font-weight:700;color:#f1f5f9;}.most-det-count{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;margin-left:auto;}
.cab{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.cab-name{font-size:11px;color:#9ca3af;width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.cab-bg{flex:1;height:16px;background:#1a1d2e;border-radius:3px;overflow:hidden;}
.cab-fill{height:16px;border-radius:3px;display:flex;align-items:center;padding-left:6px;}
.cab-pct{font-size:9px;color:#fff;font-family:'JetBrains Mono',monospace;font-weight:700;}
.slabel{font-size:9px;font-weight:700;color:#374151;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;}
.ptime{display:inline-flex;align-items:center;gap:6px;background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.18);border-radius:20px;padding:4px 12px;font-size:11px;color:#22c55e;font-family:'JetBrains Mono',monospace;margin-bottom:12px;}
.ba-label{font-size:10px;color:#4b5563;text-align:center;margin-bottom:4px;font-weight:600;letter-spacing:0.5px;}
@media(max-width:768px){
  .navbar{height:auto;flex-wrap:wrap;padding:10px 14px;gap:8px;}
  .nav-right{width:100%;justify-content:space-between;}
  .model-sw{flex:1;}.model-btn{flex:1;text-align:center;}
  .wrap{padding:12px 14px;padding-bottom:80px;}
  .mobile-nav{display:flex!important;position:fixed;bottom:0;left:0;right:0;background:#0d0f18;border-top:1px solid #1a1d2e;z-index:200;padding:8px 0 12px;}
  .mn-btn{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;background:none;border:none;cursor:pointer;font-family:'Inter',sans-serif;color:#4b5563;font-size:10px;font-weight:500;padding:4px;}
  .mn-btn.on{color:#e24b4a;}.mn-icon{font-size:20px;}
  .sv{font-size:22px;}.pred-name{font-size:17px;}.conf-big{font-size:40px;}.live-conf{font-size:48px;}.live-name{font-size:18px;}.sg{grid-template-columns:1fr 1fr;}.bn{width:110px;}
}
@media(min-width:769px){.mobile-nav{display:none!important;}}
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

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in [("tab",0),("mk","gtsrb"),("upload_history",[]),("webcam_history",[]),("live_result",None),("freeze",False),("sparkline",[])]:
    if k not in st.session_state: st.session_state[k]=v

models_dict=load_models(); all_labels=load_all_labels(); summary=load_summary()
mk=st.session_state.mk; model,mok=models_dict[mk]; labels=all_labels[mk]
both_ok=models_dict["indian"][1] and models_dict["gtsrb"][1]
mk_label="🇮🇳 Indian (59 classes)" if mk=="indian" else "🌍 GTSRB (43 classes)"

st.markdown(CSS,unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🚦</div>
    <div><div class="nav-brand">TrafficSense<span>AI</span></div><div class="nav-sub">Sign Recognition · ADAS · EfficientNet-B3</div></div>
  </div>
  <div class="nav-right">
    <div class="model-sw">
      <button class="model-btn {'active' if mk=='indian' else ''}" onclick="void(0)">🇮🇳 Indian</button>
      <button class="model-btn {'active' if mk=='gtsrb' else ''}" onclick="void(0)">🌍 GTSRB</button>
    </div>
    <div class="status-pill"><div class="sdot"></div>{'Both Models Ready' if both_ok else 'Partial Load'}</div>
  </div>
</div>""",unsafe_allow_html=True)

# ── Buttons row ────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7=st.columns([1,1,1,1,1,1,0.5])
with c1:
    if st.button("🇮🇳 Indian",key="bi",type="primary" if mk=="indian" else "secondary"):
        st.session_state.mk="indian";st.rerun()
with c2:
    if st.button("🌍 GTSRB",key="bg",type="primary" if mk=="gtsrb" else "secondary"):
        st.session_state.mk="gtsrb";st.rerun()
with c3:
    if st.button("📤 Upload",key="bu",type="primary" if st.session_state.tab==0 else "secondary"):
        st.session_state.tab=0;st.rerun()
with c4:
    if st.button("📷 Live",key="bl",type="primary" if st.session_state.tab==1 else "secondary"):
        st.session_state.tab=1;st.rerun()
with c5:
    if st.button("📊 Stats",key="bs",type="primary" if st.session_state.tab==2 else "secondary"):
        st.session_state.tab=2;st.rerun()
with c6:
    if st.button("🕒 History",key="bh",type="primary" if st.session_state.tab==3 else "secondary"):
        st.session_state.tab=3;st.rerun()
with c7:
    if st.button("◐",key="bhc",type="secondary"): pass

st.markdown("<div style='height:2px;background:#1a1d2e;'></div>",unsafe_allow_html=True)

# Mobile nav
mob="<div class='mobile-nav'>"
for icon,name,idx in [("📤","Upload",0),("📷","Live",1),("📊","Stats",2),("🕒","History",3)]:
    cls="mn-btn on" if st.session_state.tab==idx else "mn-btn"
    mob+=f'<button class="{cls}" onclick="void(0)"><span class="mn-icon">{icon}</span>{name}</button>'
mob+="</div>"
st.markdown(mob,unsafe_allow_html=True)

mk=st.session_state.mk; model,mok=models_dict[mk]; labels=all_labels[mk]
mk_label="🇮🇳 Indian (59 classes)" if mk=="indian" else "🌍 GTSRB (43 classes)"
if not mok: st.error(f"❌ {mk_label} weights not found. Place the .pth file alongside app.py.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def badge(danger):
    icon=DANGER_ICONS.get(danger,"🟢")
    cls={"MANDATORY":"bM","WARNING":"bW","INFO":"bI"}.get(danger,"bI")
    return f'<span class="badge {cls}">{icon} {danger}</span>'

def conf_ind(c):
    if c>=HIGH_CONF: return '<div class="ci-h">✅ High Confidence — reliable prediction</div>'
    if NO_SIGN_THRESHOLD<=c<LOW_CONF: return '<div class="ci-l">⚠️ Low confidence — try better lighting or crop tighter</div>'
    return ""

def render_warns(ws):
    for icon,msg in ws:
        st.markdown(f'<div class="wbanner"><span>{icon}</span><span>{msg}</span></div>',unsafe_allow_html=True)

def render_pred(cid,conf,mk_key):
    name=labels.get(str(cid),f"Class {cid}"); info=get_info(mk_key,cid)
    color=DANGER_COLORS.get(info["danger"],"#22c55e"); pct=conf*100
    st.markdown(f"""
<div class="card"><div class="ct"></div><div class="card-body">
  <div class="pred-hdr"><div><div class="pred-name">{name}</div><div class="pred-sub">{mk_label} · Class #{cid}</div></div>{badge(info['danger'])}</div>
  {conf_ind(conf)}
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#4b5563;margin-bottom:6px;"><span>Confidence</span><span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#f0f0f0;">{pct:.1f}%</span></div>
  <div class="pb-bg"><div class="pb-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div></div>
  <div class="conf-big" style="color:{color};">{pct:.1f}%</div>
  <div class="mbox"><div class="blabel" style="color:#e24b4a;">What this sign means</div><div class="btext" style="color:#d1d5db;">{info['meaning']}</div></div>
  <div class="abox"><div class="blabel" style="color:#22c55e;">ADAS Action</div><div class="btext" style="color:#86efac;">{info['action']}</div></div>
</div></div>""",unsafe_allow_html=True)

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
</div>""",unsafe_allow_html=True)

def render_top3(ids,probs,lbls):
    above20=sum(1 for p in probs[:3] if p>0.20)
    dangers=[get_info(mk,int(ids[i]))["danger"] for i in range(3) if probs[i]>0.20]
    multi=above20>=3 and len(set(dangers))>=2
    colors=["#e24b4a","#3b82f6","#7c3aed"]
    st.markdown('<div class="card"><div class="card-body"><div class="slabel">Top 3 Predictions</div>',unsafe_allow_html=True)
    for idx,prob,color in zip(ids[:3],probs[:3],colors):
        name=lbls.get(str(idx),f"Class {idx}"); pct=prob*100
        icon=DANGER_ICONS.get(get_info(mk,int(idx))["danger"],"🟢")
        st.markdown(f'<div class="br"><span class="bn">{icon} {name}</span><div class="bb"><div class="bf" style="width:{min(pct,100):.1f}%;background:{color};"></div></div><span class="bp">{pct:.1f}%</span></div>',unsafe_allow_html=True)
    if multi: st.markdown('<div class="mwarn">⚠️ Multiple sign types detected — more than one sign may be in frame</div>',unsafe_allow_html=True)
    st.markdown('</div></div>',unsafe_allow_html=True)

def draw_sparkline(vals):
    if len(vals)<2: return None
    fig,ax=plt.subplots(figsize=(4,0.9)); fig.patch.set_facecolor("#080a10"); ax.set_facecolor("#080a10")
    x=list(range(len(vals))); ax.plot(x,vals,color="#e24b4a",lw=1.5,marker="o",markersize=3)
    ax.fill_between(x,vals,alpha=0.1,color="#e24b4a"); ax.set_xlim(0,max(len(vals)-1,1)); ax.set_ylim(0,1); ax.axis("off")
    plt.tight_layout(pad=0); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=100,bbox_inches="tight",facecolor="#080a10"); buf.seek(0); plt.close(); return buf

def draw_gauge(pct,color,label):
    fig,ax=plt.subplots(figsize=(2.6,1.5),subplot_kw=dict(projection="polar")); fig.patch.set_facecolor("#0d0f18")
    theta_val=np.pi-(np.pi*pct/100)
    thetas=np.linspace(np.pi,0,100); ax.plot(thetas,[1]*100,color="#1a1d2e",lw=12,solid_capstyle="round")
    thetas2=np.linspace(np.pi,theta_val,100); ax.plot(thetas2,[1]*100,color=color,lw=12,solid_capstyle="round")
    ax.text(0,0,f"{pct:.1f}%",ha="center",va="center",fontsize=13,fontweight="700",color=color,fontfamily="monospace")
    ax.text(0,-0.55,label,ha="center",va="center",fontsize=7,color="#4b5563"); ax.set_ylim(0,1.3); ax.axis("off")
    plt.tight_layout(pad=0); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor="#0d0f18"); buf.seek(0); plt.close(); return buf

def make_csv(rows,cols):
    buf=io.StringIO(); w=csv.writer(buf); w.writerow(cols)
    for r in rows: w.writerow(r)
    return buf.getvalue().encode()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.tab==0:
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    cl,cr=st.columns([1,1],gap="large")
    with cl:
        st.markdown(f'<div class="slabel">Active Model — {mk_label}</div>',unsafe_allow_html=True)
        uploaded_files=st.file_uploader("Upload",type=["jpg","jpeg","png","webp","bmp","tiff","tif"],accept_multiple_files=True,label_visibility="collapsed")
        if not uploaded_files:
            st.markdown("""<div class="dz"><div class="dz-icon">🚦</div><div class="dz-title">Drop traffic sign images here</div><div class="dz-sub">or click Browse files above — up to 5 images at once</div><div class="dz-fmt">PNG · JPG · JPEG · WEBP · BMP · TIFF</div></div>""",unsafe_allow_html=True)
        else:
            files=uploaded_files[:5]
            if len(files)==1:
                f=files[0]; raw=Image.open(f)
                proc,warns,changed=preprocess_image(raw)
                if changed:
                    st.markdown('<div class="slabel">Preprocessing Preview</div>',unsafe_allow_html=True)
                    b1,b2=st.columns(2)
                    with b1: st.markdown('<div class="ba-label">BEFORE</div>',unsafe_allow_html=True); st.image(raw.convert("RGB").resize((224,224)),use_column_width=True)
                    with b2: st.markdown('<div class="ba-label">AFTER</div>',unsafe_allow_html=True); st.image(proc,use_column_width=True)
                else: st.image(proc,use_column_width=True)
                render_warns(warns)
                if mok:
                    with st.spinner("Classifying…"): ids,probs,ms=predict(model,proc)
                    st.markdown(f'<div class="ptime">⚡ {ms:.0f} ms · {mk_label}</div>',unsafe_allow_html=True)
                    if probs[0]<NO_SIGN_THRESHOLD: render_no_sign(probs[0])
                    else:
                        render_pred(ids[0],probs[0],mk); render_top3(ids,probs,labels)
                        ts=datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.upload_history.insert(0,{"name":labels.get(str(ids[0]),f"Class {ids[0]}")[:30],"conf":round(probs[0]*100,1),"danger":get_info(mk,int(ids[0]))["danger"],"model":mk,"time":ts,"top2":labels.get(str(ids[1]),""),"top3":labels.get(str(ids[2]),"")})
                        st.session_state.upload_history=st.session_state.upload_history[:MAX_HISTORY]
            else:
                st.markdown(f'<div class="slabel">Batch Results — {len(files)} images</div>',unsafe_allow_html=True)
                cols=st.columns(min(len(files),3))
                for i,f in enumerate(files):
                    with cols[i%3]:
                        raw=Image.open(f); proc,warns,_=preprocess_image(raw)
                        st.image(proc,use_column_width=True,caption=f.name[:18])
                        if mok:
                            with st.spinner(""): ids,probs,ms=predict(model,proc)
                            if probs[0]<NO_SIGN_THRESHOLD:
                                st.markdown('<div style="font-size:12px;color:#4b5563;text-align:center;padding:6px;">🔍 No sign</div>',unsafe_allow_html=True)
                            else:
                                name=labels.get(str(ids[0]),f"Class {ids[0]}"); info=get_info(mk,int(ids[0]))
                                color=DANGER_COLORS.get(info["danger"],"#22c55e"); icon=DANGER_ICONS.get(info["danger"],"🟢")
                                st.markdown(f'<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:8px;padding:10px;margin-top:4px;text-align:center;"><div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:4px;">{name[:20]}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;font-weight:700;color:{color};">{probs[0]*100:.1f}%</div><div style="font-size:10px;margin-top:4px;">{icon} {info["danger"]}</div></div>',unsafe_allow_html=True)
                                ts=datetime.datetime.now().strftime("%H:%M:%S")
                                st.session_state.upload_history.insert(0,{"name":name[:30],"conf":round(probs[0]*100,1),"danger":info["danger"],"model":mk,"time":ts,"top2":labels.get(str(ids[1]),""),"top3":labels.get(str(ids[2]),"")})
                st.session_state.upload_history=st.session_state.upload_history[:MAX_HISTORY]
    with cr:
        i_sum=summary.get(f"{mk}_model",summary)
        t1=i_sum.get("final_val_acc",76.81 if mk=="indian" else 83.42)
        t3=i_sum.get("final_top3_acc",92.80 if mk=="indian" else 97.09)
        n_up=len(st.session_state.upload_history)
        avg_c=np.mean([h["conf"] for h in st.session_state.upload_history]) if st.session_state.upload_history else 0.0
        st.markdown(f"""<div class="sg">
  <div class="sc"><div class="sl">Top-1 Val Acc</div><div class="sv" style="color:#22c55e;">{t1:.1f}%</div></div>
  <div class="sc"><div class="sl">Top-3 Val Acc</div><div class="sv" style="color:#3b82f6;">{t3:.1f}%</div></div>
  <div class="sc"><div class="sl">Images Classified</div><div class="sv">{n_up}</div></div>
  <div class="sc"><div class="sl">Avg Confidence</div><div class="sv" style="color:#e24b4a;">{avg_c:.1f}%</div></div>
</div>""",unsafe_allow_html=True)
        n_cls=59 if mk=="indian" else 43; ds="Indian Signs (~13K)" if mk=="indian" else "GTSRB European (50K+)"
        rows=[("Architecture","EfficientNet-B3"),("Dataset",ds),("Classes",str(n_cls)),("Backbone","ImageNet pretrained"),("Training","2-Phase Transfer Learning"),("Preprocessing","CLAHE · Bilateral · Unsharp")]
        st.markdown('<div class="slabel" style="margin-top:4px;">Model Info</div>',unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-body">',unsafe_allow_html=True)
        for k2,v2 in rows: st.markdown(f'<div class="ir"><span class="ik">{k2}</span><span class="iv">{v2}</span></div>',unsafe_allow_html=True)
        st.markdown('</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.tab==1:
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(226,75,74,0.05);border:1px solid rgba(226,75,74,0.15);border-radius:10px;padding:11px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;"><span style="font-size:20px;">📡</span><div><div style="font-size:13px;font-weight:600;color:#93c5fd;">Live Autodetect — {mk_label}</div><div style="font-size:11px;color:#4b5563;margin-top:2px;">Point at a traffic sign · Auto-capture runs continuously · Hold steady for 2 seconds 📱</div></div></div>',unsafe_allow_html=True)
    cl,cr=st.columns([1.1,1],gap="large")
    with cl:
        st.markdown('<div style="background:#0d0f18;border:1px solid #1a1d2e;border-radius:10px;padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;"><span>🎯</span><span style="font-size:11px;color:#4b5563;">Centre the sign in frame. It should fill at least 60% of the view for best results.</span></div>',unsafe_allow_html=True)
        cam=st.camera_input("Camera",label_visibility="collapsed",key=f"cam_{mk}")
        ca1,ca2=st.columns(2)
        with ca1: auto=st.checkbox("🔄 Auto-refresh",value=not st.session_state.freeze)
        with ca2:
            if st.button("❄️ Freeze" if not st.session_state.freeze else "▶️ Resume",key="frz",type="secondary"):
                st.session_state.freeze=not st.session_state.freeze;st.rerun()
        if cam and mok and not st.session_state.freeze:
            raw=Image.open(cam); proc,_,_=preprocess_image(raw)
            with st.spinner("Detecting…"): ids,probs,ms=predict(model,proc)
            no_sign=probs[0]<NO_SIGN_THRESHOLD
            st.session_state.live_result={"ids":ids,"probs":probs,"ms":ms,"no_sign":no_sign,"cid":int(ids[0]),"conf":float(probs[0])}
            if not no_sign:
                st.session_state.sparkline.append(float(probs[0])); st.session_state.sparkline=st.session_state.sparkline[-12:]
                ts=datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.webcam_history.insert(0,{"name":labels.get(str(ids[0]),f"Class {ids[0]}")[:30],"conf":round(probs[0]*100,1),"danger":get_info(mk,int(ids[0]))["danger"],"model":mk,"time":ts})
                st.session_state.webcam_history=st.session_state.webcam_history[:MAX_HISTORY]
        r=st.session_state.live_result
        if r:
            st.markdown(f'<div class="ptime">⚡ {r["ms"]:.0f} ms · {mk_label}{"  · ❄️ Frozen" if st.session_state.freeze else ""}</div>',unsafe_allow_html=True)
            if r["no_sign"]:
                st.markdown(f'<div style="background:#0d0f18;border:1px dashed #1a1d2e;border-radius:12px;padding:28px 20px;text-align:center;"><div style="font-size:34px;margin-bottom:10px;">🔍</div><div style="font-size:15px;font-weight:600;color:#4b5563;">No Sign Detected</div><div style="font-size:11px;color:#374151;margin-top:8px;">Confidence {r["conf"]*100:.1f}% — Centre the sign and hold steady</div></div>',unsafe_allow_html=True)
            else:
                cid=r["cid"]; conf=r["conf"]; name=labels.get(str(cid),f"Class {cid}")
                info=get_info(mk,cid); color=DANGER_COLORS.get(info["danger"],"#22c55e")
                st.markdown(f"""<div class="card"><div class="ct"></div><div class="card-body">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;"><div><div style="font-size:9px;color:#4b5563;letter-spacing:1px;margin-bottom:4px;">DETECTED SIGN</div><div class="live-name">{name}</div><div style="margin-top:6px;">{badge(info['danger'])}</div></div></div>
  {conf_ind(conf)}
  <div class="live-conf" style="color:{color};">{conf*100:.1f}%</div>
  <div class="abox" style="margin-top:0;"><div class="blabel" style="color:#22c55e;">ADAS Action</div><div class="btext" style="color:#86efac;">{info['action']}</div></div>
  <div class="latrow"><span>{mk_label}</span><span>Latency: {r['ms']:.0f}ms</span></div>
</div></div>""",unsafe_allow_html=True)
        if auto and cam and not st.session_state.freeze: time.sleep(0.35);st.rerun()
    with cr:
        r=st.session_state.live_result
        if r and not r["no_sign"]: render_top3(r["ids"],r["probs"],labels)
        if st.session_state.sparkline:
            st.markdown('<div class="slabel">Confidence Trend</div>',unsafe_allow_html=True)
            sp=draw_sparkline(st.session_state.sparkline)
            if sp: st.image(sp,use_column_width=True)
        wh=st.session_state.webcam_history; n_det=len(wh)
        avg_c=np.mean([h["conf"] for h in wh]) if wh else 0
        st.markdown(f"""<div class="card"><div class="card-body">
  <div class="slabel">Session Stats</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:38px;font-weight:700;color:#22c55e;">{n_det}</div>
  <div style="font-size:11px;color:#4b5563;margin-bottom:10px;">signs detected</div>
  <div style="height:4px;background:#1a1d2e;border-radius:2px;margin-bottom:10px;"><div style="width:{min(n_det*5,100)}%;height:4px;background:#22c55e;border-radius:2px;"></div></div>
  <div style="font-size:11px;color:#4b5563;">Avg confidence: <span style="color:#f0f0f0;font-family:'JetBrains Mono',monospace;">{avg_c:.1f}%</span></div>
</div></div>""",unsafe_allow_html=True)
        if st.button("🗑 Clear Webcam History",key="cwh",type="secondary"):
            st.session_state.webcam_history=[];st.session_state.live_result=None;st.session_state.sparkline=[];st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.tab==2:
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    cl,cr=st.columns([1,1],gap="large")
    i_s=summary.get("indian_model",{}); g_s=summary.get("gtsrb_model",{})
    i_t1=i_s.get("final_val_acc",76.81); i_t3=i_s.get("final_top3_acc",92.80)
    i_p1=i_s.get("phase1_best_val_acc",49.19); i_p2=i_s.get("phase2_best_val_acc",76.81)
    g_t1=g_s.get("final_val_acc",83.42); g_t3=g_s.get("final_top3_acc",97.09)
    g_p1=g_s.get("phase1_best_val_acc",39.53); g_p2=g_s.get("phase2_best_val_acc",83.42)
    with cl:
        st.markdown('<div class="slabel">Model Accuracy Gauges</div>',unsafe_allow_html=True)
        gc1,gc2=st.columns(2)
        with gc1:
            g1=draw_gauge(i_t1,"#93c5fd","🇮🇳 Indian Top-1")
            if g1: st.image(g1,use_column_width=True)
        with gc2:
            g2=draw_gauge(g_t1,"#fca5a5","🌍 GTSRB Top-1")
            if g2: st.image(g2,use_column_width=True)
        st.markdown('<div class="slabel" style="margin-top:4px;">Model Comparison</div>',unsafe_allow_html=True)
        comp=[("Top-1 Val Accuracy",f"{i_t1:.1f}%",f"{g_t1:.1f}%"),("Top-3 Val Accuracy",f"{i_t3:.1f}%",f"{g_t3:.1f}%"),("Phase 1 Best",f"{i_p1:.1f}%",f"{g_p1:.1f}%"),("Phase 2 Best",f"{i_p2:.1f}%",f"{g_p2:.1f}%"),("Classes","59","43"),("Training Images","~13K","50K+"),("Architecture","EfficientNet-B3","EfficientNet-B3")]
        rows_h="".join(f'<tr><td class="met">{m}</td><td class="vv" style="color:#93c5fd;">{iv}</td><td class="vv" style="color:#fca5a5;">{gv}</td></tr>' for m,iv,gv in comp)
        st.markdown(f'<div class="card"><div class="card-body"><table class="comptbl"><thead><tr><th>Metric</th><th style="text-align:center;">🇮🇳 Indian</th><th style="text-align:center;">🌍 GTSRB</th></tr></thead><tbody>{rows_h}</tbody></table></div></div>',unsafe_allow_html=True)
        st.markdown('<div class="slabel" style="margin-top:4px;">Training Accuracy Curves</div>',unsafe_allow_html=True)
        ih1=i_s.get("training_history",{}).get("phase1",{}); ih2=i_s.get("training_history",{}).get("phase2",{})
        gh1=g_s.get("training_history",{}).get("phase1",{}); gh2=g_s.get("training_history",{}).get("phase2",{})
        i_val=ih1.get("val_acc",[])+ih2.get("val_acc",[]); g_val=gh1.get("val_acc",[])+gh2.get("val_acc",[])
        if not i_val: i_val=[28,31,37,35,41,37,41,43,41,45,45,45,43,42,46,45,48,49,48,49,49,48,47,48,47,57,61,65,67,70,69,71,71,73,74,75,75,75,76,76,76,76,76]
        if not g_val: g_val=[19,25,30,28,34,30,34,34,33,36,34,36,37,37,37,38,38,38,38,38,38,38,39,39,39,49,54,58,61,65,68,71,73,75,77,79,80,81,81,82,81,82,83]
        p1i=len(ih1.get("val_acc",i_val[:25]))
        fig,ax=plt.subplots(figsize=(7,3.2)); fig.patch.set_facecolor("#0d0f18"); ax.set_facecolor("#080a10")
        ax.axvspan(0.5,p1i+0.5,alpha=0.04,color="#3b82f6"); ax.axvline(p1i+0.5,color="#1a1d2e",linestyle=":",lw=1)
        ax.plot(range(1,len(i_val)+1),i_val,color="#93c5fd",lw=2,label="🇮🇳 Indian",marker="o",markersize=2)
        ax.plot(range(1,len(g_val)+1),g_val,color="#fca5a5",lw=2,label="🌍 GTSRB",marker="o",markersize=2)
        ax.text(p1i/2,min(min(i_val),min(g_val))-5,"Phase 1",ha="center",fontsize=8,color="#3b82f6",style="italic")
        ax.text(p1i+(len(i_val)-p1i)/2,min(min(i_val),min(g_val))-5,"Phase 2",ha="center",fontsize=8,color="#f59e0b",style="italic")
        ax.set_xlabel("Epoch",fontsize=9,color="#4b5563"); ax.set_ylabel("Val Acc (%)",fontsize=9,color="#4b5563")
        ax.tick_params(colors="#4b5563",labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor("#1a1d2e")
        ax.grid(True,color="#1a1d2e",alpha=0.7,lw=0.5); ax.legend(fontsize=9,labelcolor="#d1d5db",facecolor="#0d0f18",edgecolor="#1a1d2e")
        plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=130,bbox_inches="tight",facecolor="#0d0f18"); buf.seek(0); plt.close()
        st.image(buf,use_column_width=True)
        st.markdown('<div class="slabel" style="margin-top:4px;">Phase Training Time Breakdown</div>',unsafe_allow_html=True)
        p1i_ep=i_s.get("phase1_epochs",25); p2i_ep=i_s.get("phase2_epochs",20)
        p1g_ep=g_s.get("phase1_epochs",25); p2g_ep=g_s.get("phase2_epochs",20)
        p1_m=[p1i_ep*108/60,p1g_ep*300/60]; p2_m=[p2i_ep*149/60,p2g_ep*416/60]
        fig2,ax2=plt.subplots(figsize=(7,1.8)); fig2.patch.set_facecolor("#0d0f18"); ax2.set_facecolor("#080a10")
        x=np.arange(2); ax2.barh(x,p1_m,color="#3b82f6",label="Phase 1",height=0.4); ax2.barh(x,p2_m,left=p1_m,color="#e24b4a",label="Phase 2",height=0.4)
        ax2.set_yticks(x); ax2.set_yticklabels(["🇮🇳 Indian","🌍 GTSRB"],fontsize=9,color="#9ca3af")
        ax2.set_xlabel("Minutes",fontsize=9,color="#4b5563"); ax2.tick_params(colors="#4b5563",labelsize=8)
        for s in ax2.spines.values(): s.set_edgecolor("#1a1d2e")
        ax2.grid(True,axis="x",color="#1a1d2e",alpha=0.7,lw=0.5); ax2.legend(fontsize=8,labelcolor="#d1d5db",facecolor="#0d0f18",edgecolor="#1a1d2e",loc="lower right")
        plt.tight_layout(); buf2=io.BytesIO(); plt.savefig(buf2,format="png",dpi=130,bbox_inches="tight",facecolor="#0d0f18"); buf2.seek(0); plt.close()
        st.image(buf2,use_column_width=True)
    with cr:
        for m_key,m_sum,color,label_t in [("indian",i_s,"#93c5fd","🇮🇳 Indian"),("gtsrb",g_s,"#fca5a5","🌍 GTSRB")]:
            pc=m_sum.get("per_class_accuracy",{})
            if pc:
                sc=sorted(pc.items(),key=lambda x:x[1],reverse=True)
                st.markdown(f'<div class="slabel" style="margin-top:4px;">{label_t} — Top 5 Classes</div>',unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-body">',unsafe_allow_html=True)
                for name,acc in sc[:5]:
                    st.markdown(f'<div class="cab"><span class="cab-name">{name[:22]}</span><div class="cab-bg"><div class="cab-fill" style="width:{min(acc,100):.0f}%;background:{color};"><span class="cab-pct">{acc:.0f}%</span></div></div></div>',unsafe_allow_html=True)
                st.markdown('</div></div>',unsafe_allow_html=True)
                st.markdown(f'<div class="slabel" style="margin-top:2px;">{label_t} — Bottom 5 Classes</div>',unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-body">',unsafe_allow_html=True)
                for name,acc in sc[-5:]:
                    st.markdown(f'<div class="cab"><span class="cab-name">{name[:22]}</span><div class="cab-bg"><div class="cab-fill" style="width:{min(acc,100):.0f}%;background:#e24b4a;"><span class="cab-pct">{acc:.0f}%</span></div></div></div>',unsafe_allow_html=True)
                st.markdown('</div></div>',unsafe_allow_html=True)
            else:
                t1_=i_t1 if m_key=="indian" else g_t1; t3_=i_t3 if m_key=="indian" else g_t3
                p1_=i_p1 if m_key=="indian" else g_p1; p2_=i_p2 if m_key=="indian" else g_p2
                n_=59 if m_key=="indian" else 43; ds_="Indian Signs (~13K)" if m_key=="indian" else "GTSRB (50K+)"
                st.markdown(f'<div class="slabel" style="margin-top:4px;">{label_t}</div>',unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-body">',unsafe_allow_html=True)
                for k2,v2 in [("Architecture","EfficientNet-B3"),("Dataset",ds_),("Classes",str(n_)),("Top-1 Acc",f"{t1_:.2f}%"),("Top-3 Acc",f"{t3_:.2f}%"),("Phase 1 Best",f"{p1_:.2f}%"),("Phase 2 Best",f"{p2_:.2f}%")]:
                    vc=color if "Acc" in k2 or "Best" in k2 else "#f0f0f0"
                    st.markdown(f'<div class="ir"><span class="ik">{k2}</span><span class="iv" style="color:{vc};">{v2}</span></div>',unsafe_allow_html=True)
                st.markdown('</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="slabel" style="margin-top:2px;">Shared Config</div>',unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-body">',unsafe_allow_html=True)
        for k2,v2 in [("Framework","PyTorch + torchvision"),("Optimiser","AdamW"),("Scheduler","CosineAnnealingLR"),("Regularisation","Mixup + Label Smoothing"),("Preprocessing","CLAHE · Bilateral · NLM · Unsharp"),("Hardware","Kaggle P100 GPU")]:
            st.markdown(f'<div class="ir"><span class="ik">{k2}</span><span class="iv">{v2}</span></div>',unsafe_allow_html=True)
        st.markdown('</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.tab==3:
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    uh=st.session_state.upload_history; wh=st.session_state.webcam_history
    all_h=uh+wh
    if all_h:
        from collections import Counter
        counts=Counter(h["name"] for h in all_h); top_sign,top_count=counts.most_common(1)[0]
        top_info=next((h for h in all_h if h["name"]==top_sign),{})
        icon=DANGER_ICONS.get(top_info.get("danger","INFO"),"🟢"); color=DANGER_COLORS.get(top_info.get("danger","INFO"),"#22c55e")
        st.markdown(f'<div class="most-det"><div class="most-det-icon">{icon}</div><div><div class="most-det-label">Most Detected Sign This Session</div><div class="most-det-name">{top_sign}</div></div><div class="most-det-count" style="color:{color};">{top_count}×</div></div>',unsafe_allow_html=True)
    dc={"MANDATORY":"#fca5a5","WARNING":"#fcd34d","INFO":"#86efac"}
    st.markdown('<div class="slabel">📤 Upload History</div>',unsafe_allow_html=True)
    if uh:
        fc1,fc2,fc3=st.columns([2,1,1])
        with fc1: su=st.text_input("Search",placeholder="Filter by name…",key="su",label_visibility="collapsed")
        with fc2: fdu=st.selectbox("Type",["All","MANDATORY","WARNING","INFO"],key="fdu",label_visibility="collapsed")
        with fc3: fmu=st.selectbox("Model",["All","indian","gtsrb"],key="fmu",label_visibility="collapsed")
        fu=[h for h in uh if (su.lower() in h["name"].lower() if su else True) and (h["danger"]==fdu if fdu!="All" else True) and (h["model"]==fmu if fmu!="All" else True)]
        st.markdown('<div class="card"><div class="card-body"><table class="htbl"><thead><tr><th>Sign</th><th>Confidence</th><th>Type</th><th>Model</th><th>Time</th></tr></thead><tbody>',unsafe_allow_html=True)
        for h in fu[:50]:
            m_="🇮🇳" if h["model"]=="indian" else "🌍"; c_=dc.get(h["danger"],"#86efac"); ico=DANGER_ICONS.get(h["danger"],"🟢")
            st.markdown(f'<tr><td class="hname">{h["name"]}</td><td class="hconf" style="color:{c_};">{h["conf"]:.1f}%</td><td>{ico} {h["danger"]}</td><td>{m_} {h["model"].upper()}</td><td style="color:#374151;font-family:\'JetBrains Mono\',monospace;font-size:10px;">{h["time"]}</td></tr>',unsafe_allow_html=True)
        st.markdown('</tbody></table></div></div>',unsafe_allow_html=True)
        if len(fu)>50: st.markdown(f'<div style="font-size:11px;color:#374151;margin-top:-10px;margin-bottom:10px;">Showing 50 of {len(fu)} results</div>',unsafe_allow_html=True)
        csv_u=make_csv([[h["name"],h["conf"],h["danger"],h["model"],h["time"],h.get("top2",""),h.get("top3","")] for h in uh],["Sign","Confidence%","Type","Model","Time","Top2","Top3"])
        st.download_button("⬇️ Export Upload History CSV",csv_u,"upload_history.csv","text/csv",key="dlu")
    else:
        st.markdown('<div class="card"><div class="card-body" style="text-align:center;padding:28px;color:#374151;">No upload history yet</div></div>',unsafe_allow_html=True)
    st.markdown("<div style='height:18px;'></div>",unsafe_allow_html=True)
    st.markdown('<div class="slabel">📷 Webcam Detection History</div>',unsafe_allow_html=True)
    if wh:
        fc1b,fc2b,fc3b=st.columns([2,1,1])
        with fc1b: sw=st.text_input("Search",placeholder="Filter by name…",key="sw",label_visibility="collapsed")
        with fc2b: fdw=st.selectbox("Type",["All","MANDATORY","WARNING","INFO"],key="fdw",label_visibility="collapsed")
        with fc3b: fmw=st.selectbox("Model",["All","indian","gtsrb"],key="fmw",label_visibility="collapsed")
        fw=[h for h in wh if (sw.lower() in h["name"].lower() if sw else True) and (h["danger"]==fdw if fdw!="All" else True) and (h["model"]==fmw if fmw!="All" else True)]
        st.markdown('<div class="card"><div class="card-body"><table class="htbl"><thead><tr><th>Sign</th><th>Confidence</th><th>Type</th><th>Model</th><th>Time</th></tr></thead><tbody>',unsafe_allow_html=True)
        for h in fw[:50]:
            m_="🇮🇳" if h["model"]=="indian" else "🌍"; c_=dc.get(h["danger"],"#86efac"); ico=DANGER_ICONS.get(h["danger"],"🟢")
            st.markdown(f'<tr><td class="hname">{h["name"]}</td><td class="hconf" style="color:{c_};">{h["conf"]:.1f}%</td><td>{ico} {h["danger"]}</td><td>{m_} {h["model"].upper()}</td><td style="color:#374151;font-family:\'JetBrains Mono\',monospace;font-size:10px;">{h["time"]}</td></tr>',unsafe_allow_html=True)
        st.markdown('</tbody></table></div></div>',unsafe_allow_html=True)
        csv_w=make_csv([[h["name"],h["conf"],h["danger"],h["model"],h["time"]] for h in wh],["Sign","Confidence%","Type","Model","Time"])
        st.download_button("⬇️ Export Webcam History CSV",csv_w,"webcam_history.csv","text/csv",key="dlw")
        if st.button("🗑 Clear All History",key="clrall",type="secondary"):
            st.session_state.upload_history=[];st.session_state.webcam_history=[];st.session_state.live_result=None;st.session_state.sparkline=[];st.rerun()
    else:
        st.markdown('<div class="card"><div class="card-body" style="text-align:center;padding:28px;color:#374151;">No webcam detections yet — only confident detections (&gt;30%) are logged</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div style="border-top:1px solid #1a1d2e;margin-top:32px;padding:14px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;"><span style="font-size:12px;color:#374151;">TrafficSense AI — Advanced Driver Assistance System</span><span style="font-size:11px;color:#374151;font-family:\'JetBrains Mono\',monospace;">EfficientNet-B3 · PyTorch · Streamlit</span></div>',unsafe_allow_html=True)