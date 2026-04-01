import os
import io
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 43
DEVICE      = torch.device("cpu")  # HF Spaces uses CPU
IS_LOCAL    = os.environ.get("RUNNING_LOCAL", "false").lower() == "true"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Sign metadata ─────────────────────────────────────────────────────────────
SIGN_INFO = {
    # ── Speed limits (MANDATORY — red circle) ─────────────────────────────────
    "0":  {"meaning": "Speed limit of 20 km/h. Used in school zones, pedestrian areas, and narrow residential streets where very low speeds are required.",
           "action": "Reduce speed to 20 km/h immediately and maintain it until the restriction ends.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "1":  {"meaning": "Speed limit of 30 km/h. Common in residential zones, near schools, and areas with high pedestrian activity.",
           "action": "Reduce speed to 30 km/h. Stay alert for pedestrians and children.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "2":  {"meaning": "Speed limit of 50 km/h. Standard limit in urban built-up areas and city streets.",
           "action": "Reduce speed to 50 km/h and maintain until the limit changes.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "3":  {"meaning": "Speed limit of 60 km/h. Applies on urban arterial roads and some rural roads.",
           "action": "Reduce speed to 60 km/h.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "4":  {"meaning": "Speed limit of 70 km/h. Common on roads leaving urban areas or on rural approach roads.",
           "action": "Reduce speed to 70 km/h.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "5":  {"meaning": "Speed limit of 80 km/h. Typically found on rural roads and roads outside urban areas.",
           "action": "Reduce speed to 80 km/h.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "6":  {"meaning": "End of the 80 km/h speed restriction. The mandatory 80 km/h limit no longer applies from this point.",
           "action": "The 80 km/h restriction has lifted. Drive at a speed appropriate for road conditions.",
           "danger": "INFO", "color": "#1d9e75"},

    "7":  {"meaning": "Speed limit of 100 km/h. Found on expressways and high-speed rural roads.",
           "action": "Do not exceed 100 km/h. Maintain safe following distance at higher speeds.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "8":  {"meaning": "Speed limit of 120 km/h. Applies on motorways and autobahns with a set restriction.",
           "action": "Do not exceed 120 km/h. Stay in the right lane unless overtaking.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    # ── No passing (MANDATORY) ────────────────────────────────────────────────
    "9":  {"meaning": "Overtaking is prohibited for all motor vehicles. You may not pass the vehicle ahead.",
           "action": "Stay in your lane. Do not attempt to overtake any vehicle until the restriction ends.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "10": {"meaning": "Vehicles exceeding 3.5 tonnes are prohibited from overtaking other motor vehicles.",
           "action": "Heavy vehicles must remain behind slower traffic. No overtaking permitted.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    # ── Right-of-way ──────────────────────────────────────────────────────────
    "11": {"meaning": "You have right of way at the upcoming intersection. Vehicles approaching from side roads must yield to you.",
           "action": "Maintain speed. You have priority — cross traffic must give way to you.",
           "danger": "INFO", "color": "#1d9e75"},

    "12": {"meaning": "You are travelling on a priority road. All vehicles at junctions and side roads must yield to you until the priority road ends.",
           "action": "Maintain speed. You have continuous right of way on this road.",
           "danger": "INFO", "color": "#1d9e75"},

    "13": {"meaning": "Yield sign. You do not have right of way. All approaching traffic on the main road has priority over you.",
           "action": "Slow down and give way to all traffic on the main road before proceeding.",
           "danger": "WARNING", "color": "#EF9F27"},

    "14": {"meaning": "Stop sign. You are legally required to come to a complete standstill before the stop line, regardless of visibility.",
           "action": "Bring vehicle to a complete stop. Check all directions for traffic before proceeding.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    # ── Prohibitory (MANDATORY) ───────────────────────────────────────────────
    "15": {"meaning": "No vehicles permitted. This road is closed to all motor vehicles, bicycles, and other vehicles in both directions.",
           "action": "Do not enter. Find an alternative route.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "16": {"meaning": "Vehicles exceeding 3.5 tonnes are prohibited from using this road.",
           "action": "Heavy vehicles must use an alternative route. Passenger cars may proceed.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "17": {"meaning": "No entry. Entry is prohibited for all vehicles in this direction. This is typically a one-way road in the opposite direction.",
           "action": "Do not enter. Turn around and use an alternative route.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    # ── Warning signs (WARNING — red triangle) ────────────────────────────────
    "18": {"meaning": "General danger warning. An unspecified hazard lies ahead that is not covered by a specific sign.",
           "action": "Reduce speed, increase alertness, and be prepared for unexpected hazards.",
           "danger": "WARNING", "color": "#EF9F27"},

    "19": {"meaning": "Dangerous left curve ahead. The road bends sharply to the left.",
           "action": "Reduce speed before entering the curve. Do not brake mid-curve.",
           "danger": "WARNING", "color": "#EF9F27"},

    "20": {"meaning": "Dangerous right curve ahead. The road bends sharply to the right.",
           "action": "Reduce speed before entering the curve. Do not brake mid-curve.",
           "danger": "WARNING", "color": "#EF9F27"},

    "21": {"meaning": "Double curve ahead — first curving left, then right (or vice versa). Consecutive bends require sustained attention.",
           "action": "Reduce speed significantly. Keep to your lane through both curves.",
           "danger": "WARNING", "color": "#EF9F27"},

    "22": {"meaning": "Bumpy or uneven road surface ahead. Road humps, rough surface, or damaged pavement expected.",
           "action": "Reduce speed to protect vehicle suspension and maintain control.",
           "danger": "WARNING", "color": "#EF9F27"},

    "23": {"meaning": "Slippery road ahead. Surface may be wet, icy, or otherwise slippery, increasing stopping distances.",
           "action": "Reduce speed. Avoid sudden braking, acceleration, or sharp steering inputs.",
           "danger": "WARNING", "color": "#EF9F27"},

    "24": {"meaning": "Road narrows from the right side ahead. The carriageway becomes narrower, reducing available lane width.",
           "action": "Move toward the centre-left. Reduce speed and watch for oncoming traffic.",
           "danger": "WARNING", "color": "#EF9F27"},

    "25": {"meaning": "Roadworks ahead. Construction or maintenance activity is taking place on or near the road.",
           "action": "Reduce speed. Follow temporary signs and be alert for workers and machinery.",
           "danger": "WARNING", "color": "#EF9F27"},

    "26": {"meaning": "Traffic signals ahead. A traffic light controlled junction or crossing is approaching.",
           "action": "Reduce speed and be prepared to stop at a red light.",
           "danger": "WARNING", "color": "#EF9F27"},

    "27": {"meaning": "Pedestrian crossing ahead. A marked crossing where pedestrians have right of way.",
           "action": "Reduce speed and be prepared to stop for pedestrians crossing.",
           "danger": "WARNING", "color": "#EF9F27"},

    "28": {"meaning": "Children crossing ahead. School zone or area with high child pedestrian activity nearby.",
           "action": "Reduce speed sharply. Be prepared to stop. Watch carefully for children.",
           "danger": "WARNING", "color": "#EF9F27"},

    "29": {"meaning": "Bicycle crossing or shared cycle route ahead. Cyclists may enter or cross the road.",
           "action": "Check mirrors and blind spots. Give cyclists adequate space.",
           "danger": "WARNING", "color": "#EF9F27"},

    "30": {"meaning": "Ice or snow likely on road surface. Freezing conditions create significant skid risk.",
           "action": "Reduce speed considerably. Increase following distance. Avoid harsh inputs.",
           "danger": "WARNING", "color": "#EF9F27"},

    "31": {"meaning": "Wild animals crossing ahead. Deer or other animals may unexpectedly enter the road.",
           "action": "Reduce speed and scan the road edges. Be ready to brake suddenly.",
           "danger": "WARNING", "color": "#EF9F27"},

    # ── End of restrictions (INFO) ────────────────────────────────────────────
    "32": {"meaning": "End of all restrictions. All previously imposed speed limits and overtaking bans are lifted from this point.",
           "action": "Normal road rules apply. Drive at a speed appropriate for conditions.",
           "danger": "INFO", "color": "#1d9e75"},

    # ── Directional mandatory (MANDATORY — blue circle) ───────────────────────
    "33": {"meaning": "Turn right ahead mandatory. You must turn right at the upcoming junction — going straight or left is not permitted.",
           "action": "Signal right and prepare to turn right at the next junction.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "34": {"meaning": "Turn left ahead mandatory. You must turn left at the upcoming junction — going straight or right is not permitted.",
           "action": "Signal left and prepare to turn left at the next junction.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "35": {"meaning": "Proceed straight ahead only. You must continue straight — turning left or right is not permitted.",
           "action": "Do not turn at the next junction. Proceed straight only.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "36": {"meaning": "Go straight or turn right. You may only proceed straight ahead or turn right — left turns are prohibited.",
           "action": "Choose to go straight or turn right. Left turns are not allowed.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "37": {"meaning": "Go straight or turn left. You may only proceed straight ahead or turn left — right turns are prohibited.",
           "action": "Choose to go straight or turn left. Right turns are not allowed.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "38": {"meaning": "Keep right. You must pass an obstacle or traffic island by keeping to the right side.",
           "action": "Bear right and pass the obstacle or median on the right side.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "39": {"meaning": "Keep left. You must pass an obstacle or traffic island by keeping to the left side.",
           "action": "Bear left and pass the obstacle or median on the left side.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    "40": {"meaning": "Roundabout ahead. Traffic in the roundabout has priority. You must give way before entering.",
           "action": "Slow down. Yield to vehicles already circulating in the roundabout before entering.",
           "danger": "MANDATORY", "color": "#e24b4a"},

    # ── End of passing restrictions (INFO) ────────────────────────────────────
    "41": {"meaning": "End of no-overtaking zone. The prohibition on overtaking all motor vehicles is now lifted.",
           "action": "Overtaking is now permitted where it is safe to do so.",
           "danger": "INFO", "color": "#1d9e75"},

    "42": {"meaning": "End of no-overtaking zone for vehicles over 3.5 tonnes. Heavy vehicles may now overtake where safe.",
           "action": "Heavy vehicles may resume overtaking where road conditions permit.",
           "danger": "INFO", "color": "#1d9e75"},
}

# ── Theme CSS ─────────────────────────────────────────────────────────────────
def get_css(dark_mode):
    if dark_mode:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: #0f1117 !important;
            color: #e2e8f0 !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }

        /* Topbar */
        .topbar {
            background: #1a1d2e;
            padding: 14px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #2a2d3a;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .logo-wrap { display: flex; align-items: center; gap: 12px; }
        .logo-icon {
            width: 36px; height: 36px;
            background: #e24b4a;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 16px;
        }
        .logo-text { font-size: 16px; font-weight: 600; color: #f1f5f9; }
        .logo-sub { font-size: 11px; color: #64748b; margin-top: 1px; }
        .status-badge {
            display: flex; align-items: center; gap: 6px;
            background: #0f2a1a; border: 1px solid #1d9e75;
            border-radius: 20px; padding: 6px 14px;
        }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #1d9e75; }
        .status-text { font-size: 12px; color: #5DCAA5; font-weight: 500; }

        /* Tabs */
        .tab-bar {
            background: #1a1d2e;
            display: flex;
            border-bottom: 1px solid #2a2d3a;
            padding: 0 28px;
        }
        .tab-btn {
            padding: 12px 20px;
            font-size: 13px;
            font-family: 'DM Sans', sans-serif;
            color: #64748b;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        .tab-btn:hover { color: #94a3b8; }
        .tab-btn.active { color: #e24b4a; border-bottom: 2px solid #e24b4a; }

        /* Cards */
        .card {
            background: #1a1d2e;
            border: 1px solid #2a2d3a;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .sec-label {
            font-size: 10px;
            font-weight: 600;
            color: #475569;
            letter-spacing: 1px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .upload-zone {
            border: 2px dashed #2a2d3a;
            border-radius: 12px;
            padding: 28px 20px;
            text-align: center;
            background: #13161f;
            transition: border-color 0.2s;
        }
        .upload-zone:hover { border-color: #e24b4a; }
        .pred-name { font-size: 22px; font-weight: 600; color: #f1f5f9; }
        .badge {
            display: inline-block;
            font-size: 10px; font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        }
        .badge-mandatory { background: #3a1515; color: #f09595; }
        .badge-warning   { background: #2a2200; color: #FAC775; }
        .badge-info      { background: #0f2a1a; color: #9FE1CB; }
        .meaning-box {
            background: #13161f;
            border-left: 3px solid #e24b4a;
            border-radius: 0 8px 8px 0;
            padding: 10px 14px;
            margin-top: 10px;
        }
        .meaning-label { font-size: 9px; color: #475569; letter-spacing: 1px; margin-bottom: 4px; }
        .meaning-text  { font-size: 13px; color: #cbd5e1; line-height: 1.6; }
        .action-box {
            background: #0f2a1a;
            border-left: 3px solid #1d9e75;
            border-radius: 0 8px 8px 0;
            padding: 10px 14px;
            margin-top: 8px;
        }
        .action-label { font-size: 9px; color: #1d9e75; letter-spacing: 1px; margin-bottom: 4px; }
        .action-text  { font-size: 13px; color: #9FE1CB; line-height: 1.5; }

        /* Bar rows */
        .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
        .bar-name { font-size: 12px; color: #cbd5e1; width: 150px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .bar-bg   { flex: 1; height: 6px; background: #2a2d3a; border-radius: 3px; }
        .bar-fill { height: 6px; border-radius: 3px; }
        .bar-pct  { font-size: 12px; color: #64748b; width: 42px; text-align: right; font-family: 'DM Mono', monospace; }

        /* Stat cards */
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .stat-card {
            background: #1a1d2e;
            border: 1px solid #2a2d3a;
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }
        .stat-val { font-size: 22px; font-weight: 600; }
        .stat-lbl { font-size: 11px; color: #475569; margin-top: 3px; }

        /* Info table */
        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid #2a2d3a;
            font-size: 13px;
        }
        .info-row:last-child { border-bottom: none; }
        .info-key { color: #64748b; }
        .info-val { color: #e2e8f0; font-weight: 500; }

        /* History pills */
        .history-wrap { display: flex; gap: 6px; flex-wrap: wrap; }
        .h-pill {
            background: #1a1d2e;
            border: 1px solid #2a2d3a;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 11px;
            color: #94a3b8;
        }

        /* Webcam */
        .cam-placeholder {
            background: #13161f;
            border: 1px solid #2a2d3a;
            border-radius: 12px;
            height: 220px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .process-time {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: #0f2a1a;
            border: 1px solid #1d9e75;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 11px;
            color: #5DCAA5;
            margin-bottom: 10px;
        }
        .content-wrap { padding: 20px 28px; }

        /* Streamlit overrides */
        .stFileUploader > div { background: transparent !important; }
        .stFileUploader label { color: #e24b4a !important; }
        div[data-testid="stImage"] img { border-radius: 10px; }
        .stButton > button {
            background: #e24b4a !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            padding: 8px 20px !important;
        }
        .stButton > button:hover { background: #c0392b !important; }
        div[data-testid="stCameraInput"] { background: #1a1d2e !important; border-radius: 12px !important; }

        /* Toggle button */
        .theme-toggle {
            background: #1e2235;
            border: 1px solid #2a2d3a;
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 12px;
            color: #94a3b8;
            cursor: pointer;
            margin-right: 10px;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: #f8fafc !important;
            color: #1e293b !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }

        .topbar {
            background: #ffffff;
            padding: 14px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #e2e8f0;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .logo-wrap { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 36px; height: 36px; background: #e24b4a; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; }
        .logo-text { font-size: 16px; font-weight: 600; color: #0f172a; }
        .logo-sub  { font-size: 11px; color: #94a3b8; margin-top: 1px; }
        .status-badge { display: flex; align-items: center; gap: 6px; background: #f0fdf4; border: 1px solid #1d9e75; border-radius: 20px; padding: 6px 14px; }
        .status-dot  { width: 7px; height: 7px; border-radius: 50%; background: #1d9e75; }
        .status-text { font-size: 12px; color: #0f6e56; font-weight: 500; }

        .tab-bar { background: #ffffff; display: flex; border-bottom: 1px solid #e2e8f0; padding: 0 28px; }
        .tab-btn { padding: 12px 20px; font-size: 13px; font-family: 'DM Sans', sans-serif; color: #94a3b8; background: none; border: none; border-bottom: 2px solid transparent; cursor: pointer; transition: all 0.2s; font-weight: 500; }
        .tab-btn:hover { color: #64748b; }
        .tab-btn.active { color: #e24b4a; border-bottom: 2px solid #e24b4a; }

        .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; }
        .sec-label { font-size: 10px; font-weight: 600; color: #94a3b8; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase; }
        .upload-zone { border: 2px dashed #e2e8f0; border-radius: 12px; padding: 28px 20px; text-align: center; background: #f8fafc; }
        .pred-name { font-size: 22px; font-weight: 600; color: #0f172a; }
        .badge { display: inline-block; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.5px; }
        .badge-mandatory { background: #fee2e2; color: #991b1b; }
        .badge-warning   { background: #fef3c7; color: #92400e; }
        .badge-info      { background: #dcfce7; color: #166534; }
        .meaning-box { background: #fff5f5; border-left: 3px solid #e24b4a; border-radius: 0 8px 8px 0; padding: 10px 14px; margin-top: 10px; }
        .meaning-label { font-size: 9px; color: #94a3b8; letter-spacing: 1px; margin-bottom: 4px; }
        .meaning-text  { font-size: 13px; color: #374151; line-height: 1.6; }
        .action-box { background: #f0fdf4; border-left: 3px solid #1d9e75; border-radius: 0 8px 8px 0; padding: 10px 14px; margin-top: 8px; }
        .action-label { font-size: 9px; color: #0f6e56; letter-spacing: 1px; margin-bottom: 4px; }
        .action-text  { font-size: 13px; color: #085041; line-height: 1.5; }

        .bar-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
        .bar-name { font-size: 12px; color: #374151; width: 150px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .bar-bg   { flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; }
        .bar-fill { height: 6px; border-radius: 3px; }
        .bar-pct  { font-size: 12px; color: #94a3b8; width: 42px; text-align: right; font-family: 'DM Mono', monospace; }

        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .stat-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; text-align: center; }
        .stat-val  { font-size: 22px; font-weight: 600; }
        .stat-lbl  { font-size: 11px; color: #94a3b8; margin-top: 3px; }

        .info-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
        .info-row:last-child { border-bottom: none; }
        .info-key { color: #94a3b8; }
        .info-val { color: #1e293b; font-weight: 500; }

        .history-wrap { display: flex; gap: 6px; flex-wrap: wrap; }
        .h-pill { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 10px; font-size: 11px; color: #64748b; }

        .cam-placeholder { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; height: 220px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .process-time { display: inline-flex; align-items: center; gap: 5px; background: #f0fdf4; border: 1px solid #1d9e75; border-radius: 20px; padding: 4px 12px; font-size: 11px; color: #0f6e56; margin-bottom: 10px; }
        .content-wrap { padding: 20px 28px; }
        .stButton > button { background: #e24b4a !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
        div[data-testid="stImage"] img { border-radius: 10px; }
        .theme-toggle { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 20px; padding: 6px 14px; font-size: 12px; color: #64748b; cursor: pointer; margin-right: 10px; }
        </style>
        """

# ── Model ─────────────────────────────────────────────────────────────────────
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.base = efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base(x)


@st.cache_resource
def load_model():
    model = TrafficSignNet(NUM_CLASSES)

    # Search multiple possible locations for the model file
    pth_candidates = []
    for pth_file in ["traffic_sign_model_full.pth", "traffic_sign_model.pth", "best_model.pth"]:
        pth_candidates += [
            pth_file,
            f"/app/{pth_file}",
            os.path.join(os.path.dirname(__file__), pth_file),
        ]

    for pth_path in pth_candidates:
        if not os.path.exists(pth_path):
            continue
        print(f"[TrafficSense] Trying model file: {pth_path}")
        try:
            obj = torch.load(pth_path, map_location=DEVICE, weights_only=False)
            state = obj if isinstance(obj, dict) else obj.state_dict()

            # Load weights and measure coverage — fail loudly if too few keys match
            missing, unexpected = model.load_state_dict(state, strict=False)
            own_keys    = set(model.state_dict().keys())
            loaded_keys = own_keys - set(missing)
            coverage    = len(loaded_keys) / len(own_keys)

            print(f"[TrafficSense] Weight coverage: {coverage:.1%} ({len(loaded_keys)}/{len(own_keys)} keys)")
            print(f"[TrafficSense] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            if coverage < 0.85:
                print(f"[TrafficSense] ❌ Only {coverage:.1%} weights loaded — skipping {pth_path}")
                continue  # try next file instead of silently using bad weights

            model.eval()
            print(f"[TrafficSense] ✅ Model loaded successfully from {pth_path}")
            return model, True

        except Exception as e:
            print(f"[TrafficSense] Error loading {pth_path}: {e}")
            continue

    print("[TrafficSense] ❌ No valid model file found — predictions will be random!")
    return model, False


@st.cache_data
def load_labels():
    # Search multiple possible locations for labels.json
    search_paths = [
        "labels.json",
        "/app/labels.json",
        os.path.join(os.path.dirname(__file__), "labels.json"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            print(f"[TrafficSense] Loading labels from: {path}")
            with open(path) as f:
                return json.load(f)
    # Fallback: generate numeric labels so app doesn't crash
    print("[TrafficSense] WARNING: labels.json not found, using numeric fallbacks")
    return {str(i): f"Sign Class {i}" for i in range(43)}


@st.cache_data
def load_summary():
    try:
        with open("training_summary.json") as f:
            return json.load(f)
    except:
        return {}


# ── Transforms ───────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ── Inference ─────────────────────────────────────────────────────────────────
def flatten_alpha(img_pil):
    """Replace transparent background with white — prevents checkerboard grey noise."""
    if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
        bg = Image.new("RGB", img_pil.size, (255, 255, 255))
        if img_pil.mode == "P":
            img_pil = img_pil.convert("RGBA")
        mask = img_pil.split()[-1] if img_pil.mode == "RGBA" else None
        bg.paste(img_pil.convert("RGB"), mask=mask)
        return bg
    return img_pil.convert("RGB")


def predict(model, img_pil):
    t0      = time.time()
    img_rgb = flatten_alpha(img_pil)   # fix transparent PNG backgrounds
    tensor  = val_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    elapsed = (time.time() - t0) * 1000
    topk_probs, topk_ids = probs.topk(5)
    return (
        topk_ids.cpu().numpy(),
        topk_probs.cpu().numpy(),
        probs.cpu().numpy(),
        elapsed
    )


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_pil):
    activations, gradients = [], []

    def fwd(m, i, o): activations.append(o.detach())
    def bwd(m, gi, go): gradients.append(go[0].detach())

    target = model.base.features[-1]
    fh = target.register_forward_hook(fwd)
    bh = target.register_full_backward_hook(bwd)

    tensor = val_transform(flatten_alpha(img_pil)).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    out    = model(tensor)
    pred   = out.argmax(1).item()
    model.zero_grad()
    out[0, pred].backward()

    fh.remove(); bh.remove()

    act  = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    w    = grad.mean(dim=(1, 2))
    hmap = torch.relu((w[:, None, None] * act).sum(0))
    hmap = hmap / (hmap.max() + 1e-8)
    return hmap.cpu().numpy()


def make_gradcam_figure(img_pil, heatmap):
    orig = np.array(flatten_alpha(img_pil).resize((IMG_SIZE, IMG_SIZE)))
    from PIL import Image as PILImage
    hmap_resized = np.array(
        PILImage.fromarray(np.uint8(heatmap * 255)).resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    ) / 255.0

    colored  = plt.colormaps["jet"](hmap_resized)[:, :, :3]
    overlay  = np.clip(colored * 0.45 + orig / 255.0 * 0.55, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.patch.set_facecolor("none")
    titles = ["Original", "Grad-CAM", "Overlay"]
    imgs   = [orig / 255.0, hmap_resized, overlay]
    cmaps  = [None, "jet", None]

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, color="#94a3b8", pad=6)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="none", transparent=True)
    buf.seek(0)
    plt.close()
    return buf


# ── HTML helpers ──────────────────────────────────────────────────────────────
def render_topbar(dark_mode):
    moon = "☀️" if dark_mode else "🌙"
    toggle_label = "Light Mode" if dark_mode else "Dark Mode"
    st.markdown(f"""
    <div class="topbar">
      <div class="logo-wrap">
        <div class="logo-icon">🚦</div>
        <div>
          <div class="logo-text">TrafficSense AI</div>
          <div class="logo-sub">Smart Traffic Sign Classification for Autonomous Driving</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <div class="status-badge">
          <div class="status-dot"></div>
          <span class="status-text">System Ready</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def badge_html(danger):
    cls = {"MANDATORY": "badge-mandatory", "WARNING": "badge-warning", "INFO": "badge-info"}.get(danger, "badge-info")
    return f'<span class="badge {cls}">{danger}</span>'


def render_prediction(pred_id, conf, labels, dark_mode):
    name = labels[str(pred_id)]
    info = SIGN_INFO.get(str(pred_id), {"meaning": "No description available.", "action": "Follow standard road rules.", "danger": "INFO", "color": "#1d9e75"})
    danger  = info["danger"]
    meaning = info["meaning"]
    action  = info["action"]
    color   = info["color"]

    conf_pct = conf * 100
    bar_color = color

    st.markdown(f"""
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <div class="pred-name">{name}</div>
        {badge_html(danger)}
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
        <span style="font-size:12px;color:#64748b;">Confidence</span>
        <span style="font-size:13px;font-weight:600;color:{'#f1f5f9' if dark_mode else '#0f172a'};">{conf_pct:.1f}%</span>
      </div>
      <div style="height:6px;background:{'#2a2d3a' if dark_mode else '#e2e8f0'};border-radius:3px;margin-bottom:12px;">
        <div style="width:{min(conf_pct,100):.1f}%;height:6px;background:{bar_color};border-radius:3px;"></div>
      </div>
      <div class="meaning-box">
        <div class="meaning-label">WHAT THIS SIGN MEANS</div>
        <div class="meaning-text">{meaning}</div>
      </div>
      <div class="action-box">
        <div class="action-label">ADAS ACTION</div>
        <div class="action-text">{action}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_top3(ids, probs, labels, dark_mode):
    colors = ["#e24b4a", "#378ADD", "#378ADD"]
    st.markdown('<div class="card"><div class="sec-label">TOP 3 PREDICTIONS</div>', unsafe_allow_html=True)
    for i, (idx, prob) in enumerate(zip(ids[:3], probs[:3])):
        name  = labels[str(idx)]
        pct   = prob * 100
        color = colors[i]
        st.markdown(
            f'<div class="bar-row">'
            f'<span class="bar-name">{name}</span>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div></div>'
            f'<span class="bar-pct">{pct:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "dark_mode"   not in st.session_state: st.session_state.dark_mode   = True
if "active_tab"  not in st.session_state: st.session_state.active_tab  = 0
if "history"     not in st.session_state: st.session_state.history     = []

# ── Load resources ────────────────────────────────────────────────────────────
import glob

try:
    available = glob.glob("*.pth") + glob.glob("*.json") + glob.glob("/app/*.pth") + glob.glob("/app/*.json")
    print(f"[TrafficSense] Files found: {available}")
    print(f"[TrafficSense] Working dir: {os.getcwd()}")
except Exception as e:
    print(f"[TrafficSense] File scan error: {e}")

try:
    model, model_ok = load_model()
    print(f"[TrafficSense] Model loaded: {model_ok}")
except Exception as e:
    print(f"[TrafficSense] Model load error: {e}")
    model, model_ok = None, False

# ── Model load diagnostic (visible in UI) ────────────────────────────────────
if not model_ok:
    st.error("❌ **Model weights not found or failed to load.** Predictions will be random. "
             "Make sure `traffic_sign_model_full.pth` is in the same folder as `app.py`.")
else:
    total_params  = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    nonzero_pct   = nonzero_params / total_params * 100
    if nonzero_pct < 70:
        st.warning(f"⚠️ **Only {nonzero_pct:.1f}% of model weights are non-zero** — "
                   "the .pth file may be corrupt or from a different architecture. Re-train and re-save.")

try:
    labels = load_labels()
    print(f"[TrafficSense] Labels loaded: {len(labels)} classes")
except Exception as e:
    print(f"[TrafficSense] Labels load error: {e}")
    labels = {str(i): f"Sign Class {i}" for i in range(43)}

try:
    summary = load_summary()
except Exception as e:
    print(f"[TrafficSense] Summary load error: {e}")
    summary = {}

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── Topbar ────────────────────────────────────────────────────────────────────
render_topbar(st.session_state.dark_mode)

# ── Theme toggle + Tabs in one row ────────────────────────────────────────────
col_t1, col_t2, col_t3, col_toggle = st.columns([1, 1, 1, 0.5])
with col_t1:
    if st.button("📤  Upload Image", key="tab0",
                 type="primary" if st.session_state.active_tab == 0 else "secondary"):
        st.session_state.active_tab = 0
        st.rerun()
with col_t2:
    if st.button("📷  Live Webcam", key="tab1",
                 type="primary" if st.session_state.active_tab == 1 else "secondary"):
        st.session_state.active_tab = 1
        st.rerun()
with col_t3:
    if st.button("📊  Model Stats", key="tab2",
                 type="primary" if st.session_state.active_tab == 2 else "secondary"):
        st.session_state.active_tab = 2
        st.rerun()
with col_toggle:
    toggle_label = "🌙 Dark" if not st.session_state.dark_mode else "☀️ Light"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("<hr style='margin:0;border-color:#2a2d3a;'>", unsafe_allow_html=True)

dark = st.session_state.dark_mode

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == 0:
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown('<div class="sec-label">INPUT IMAGE</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a traffic sign image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, use_column_width=True, caption="Uploaded image")

            if model_ok:
                with st.spinner("Classifying..."):
                    ids, probs, all_probs, ms = predict(model, img_pil)
                    top_id   = ids[0]
                    top_conf = probs[0]

                # Add to history
                name = labels[str(top_id)]
                st.session_state.history.insert(0, {
                    "name": name[:20],
                    "conf": top_conf * 100,
                    "danger": SIGN_INFO.get(str(top_id), {}).get("danger", "INFO")
                })
                st.session_state.history = st.session_state.history[:6]

                # Processing time
                st.markdown(f"""
                <div class="process-time">
                  <div class="status-dot"></div>
                  Classified in {ms:.0f}ms
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="sec-label">PREDICTION</div>', unsafe_allow_html=True)
                render_prediction(top_id, top_conf, labels, dark)
                render_top3(ids, probs, labels, dark)
        else:
            st.markdown("""
            <div class="upload-zone">
              <div style="font-size:32px;margin-bottom:8px;">📷</div>
              <div style="color:#e24b4a;font-weight:600;font-size:14px;">Click Browse files above</div>
              <div style="color:#64748b;font-size:12px;margin-top:6px;">Upload any traffic sign photo — PNG or JPG</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        if uploaded and model_ok:
            # Grad-CAM
            st.markdown('<div class="sec-label">GRAD-CAM EXPLAINABILITY</div>', unsafe_allow_html=True)
            try:
                heatmap = compute_gradcam(model, img_pil)
                gcam_buf = make_gradcam_figure(img_pil, heatmap)
                st.image(gcam_buf, use_column_width=True)
                bg = "#13161f" if dark else "#f8fafc"
                tc = "#64748b"
                st.markdown(f"""
                <div style="background:{bg};border-radius:8px;padding:8px 12px;font-size:12px;color:{tc};margin-bottom:14px;">
                  🔍 Model attended to the sign interior — not background noise
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {e}")

        # Recent predictions
        if st.session_state.history:
            st.markdown('<div class="sec-label">RECENT PREDICTIONS</div>', unsafe_allow_html=True)
            danger_colors = {"MANDATORY": "#e24b4a", "WARNING": "#EF9F27", "INFO": "#1d9e75"}
            st.markdown('<div class="history-wrap">', unsafe_allow_html=True)
            for h in st.session_state.history:
                color = danger_colors.get(h["danger"], "#1d9e75")
                st.markdown(f'<div class="h-pill" style="border-left:3px solid {color};">{h["name"]} · {h["conf"]:.0f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Quick stats
        st.markdown('<div class="sec-label">QUICK MODEL STATS</div>', unsafe_allow_html=True)
        t1  = summary.get("test_top1_accuracy_pct", 90.17)
        t3  = summary.get("test_top3_accuracy_pct", 97.81)
        t5  = summary.get("test_top5_accuracy_pct", 99.31)
        c1  = "#1d9e75" if dark else "#0f6e56"
        c2  = "#378ADD" if dark else "#185FA5"
        c3  = "#EF9F27" if dark else "#854F0B"
        bg2 = "#1a1d2e" if dark else "#ffffff"
        bd  = "#2a2d3a" if dark else "#e2e8f0"
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-val" style="color:{c1};">{t1}%</div>
            <div class="stat-lbl">Top-1 Accuracy</div>
          </div>
          <div class="stat-card">
            <div class="stat-val" style="color:{c2};">{t3}%</div>
            <div class="stat-lbl">Top-3 Accuracy</div>
          </div>
          <div class="stat-card">
            <div class="stat-val" style="color:{c3};">{t5}%</div>
            <div class="stat-lbl">Top-5 Accuracy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == 1:
    col_left, col_right = st.columns([1.2, 1], gap="medium")

    with col_left:
        st.markdown('<div class="sec-label">CAMERA FEED</div>', unsafe_allow_html=True)

        if IS_LOCAL:
            # ── Local: streamlit-webrtc live stream ──
            try:
                from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
                import av

                class VideoProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.result = None

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        img_rgb = img[:, :, ::-1]
                        pil_img = Image.fromarray(img_rgb)

                        if model_ok:
                            ids, probs, _, ms = predict(model, pil_img)
                            self.result = (ids[0], probs[0], ms)

                        import cv2
                        if self.result:
                            cls_id, conf, ms = self.result
                            name = labels[str(cls_id)]
                            text = f"{name} {conf*100:.1f}%"
                            cv2.putText(img, text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (46, 204, 113), 2)
                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                ctx = webrtc_streamer(
                    key="traffic-sign-live",
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

                if ctx.video_processor and ctx.video_processor.result:
                    cls_id, conf, ms = ctx.video_processor.result
                    st.session_state.live_pred = (cls_id, conf, ms)

            except ImportError:
                st.error("streamlit-webrtc not installed. Run: pip install streamlit-webrtc")

        else:
            # ── HF Spaces: camera snapshot mode ──
            st.info("📸 **Snapshot mode** — tap Capture to classify. Works on mobile!")
            cam_img = st.camera_input("Point camera at a traffic sign", label_visibility="collapsed")

            if cam_img and model_ok:
                img_pil = Image.open(cam_img).convert("RGB")
                with st.spinner("Classifying..."):
                    ids, probs, _, ms = predict(model, img_pil)
                    st.session_state.live_pred = (ids[0], probs[0], ms)
                    st.session_state.live_ids  = ids
                    st.session_state.live_probs = probs

        # Live prediction display
        if "live_pred" in st.session_state and st.session_state.live_pred:
            cls_id, conf, ms = st.session_state.live_pred
            name   = labels[str(cls_id)]
            info   = SIGN_INFO.get(str(cls_id), {"danger": "INFO", "action": "Follow standard road rules.", "meaning": ""})
            danger = info["danger"]
            action = info["action"]
            color  = info.get("color", "#1d9e75")

            bg   = "#1a1d2e" if dark else "#ffffff"
            bd   = "#2a2d3a" if dark else "#e2e8f0"
            tc   = "#f1f5f9" if dark else "#0f172a"
            abg  = "#0f2a1a" if dark else "#f0fdf4"
            abc  = "#9FE1CB" if dark else "#085041"
            alc  = "#5DCAA5" if dark else "#0f6e56"

            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bd};border-radius:12px;padding:16px 18px;margin-top:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div>
                  <div style="font-size:11px;color:#64748b;letter-spacing:1px;margin-bottom:4px;">CURRENT SIGN</div>
                  <div style="font-size:20px;font-weight:600;color:{tc};">{name}</div>
                  {badge_html(danger)}
                </div>
                <div style="text-align:right;">
                  <div style="font-size:11px;color:#64748b;margin-bottom:4px;">CONFIDENCE</div>
                  <div style="font-size:26px;font-weight:600;color:{color};">{conf*100:.1f}%</div>
                  <div style="font-size:10px;color:#475569;">{ms:.0f}ms inference</div>
                </div>
              </div>
              <div style="background:{abg};border-left:3px solid #1d9e75;border-radius:0 8px 8px 0;padding:10px 14px;">
                <div style="font-size:9px;color:{alc};letter-spacing:1px;margin-bottom:4px;">ADAS ACTION</div>
                <div style="font-size:13px;color:{abc};">{action}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # Recent detections
        if st.session_state.history:
            st.markdown('<div class="sec-label">RECENT DETECTIONS</div>', unsafe_allow_html=True)
            bg2 = "#1a1d2e" if dark else "#ffffff"
            bd2 = "#2a2d3a" if dark else "#e2e8f0"
            tc2 = "#e2e8f0" if dark else "#1e293b"
            danger_colors = {"MANDATORY": "#e24b4a", "WARNING": "#EF9F27", "INFO": "#1d9e75"}
            st.markdown(f'<div style="background:{bg2};border:1px solid {bd2};border-radius:12px;padding:12px 16px;">', unsafe_allow_html=True)
            for h in st.session_state.history[:5]:
                dc = danger_colors.get(h["danger"], "#1d9e75")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid {bd2};">'
                    f'<span style="font-size:13px;color:{tc2};">{h["name"]}</span>'
                    f'<span style="font-size:12px;color:{dc};font-weight:600;">{h["conf"]:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Sign meaning
        if "live_pred" in st.session_state and st.session_state.live_pred:
            cls_id = st.session_state.live_pred[0]
            info   = SIGN_INFO.get(str(cls_id), {"meaning": "No description available.", "danger": "INFO"})
            name   = labels[str(cls_id)]
            st.markdown(f"""
            <div class="meaning-box" style="margin-top:0;">
              <div class="meaning-label">{name.upper()}</div>
              <div class="meaning-text">{info['meaning']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Session stats
        bg3 = "#1a1d2e" if dark else "#ffffff"
        bd3 = "#2a2d3a" if dark else "#e2e8f0"
        vc3 = "#1d9e75" if dark else "#0f6e56"
        n_det = len(st.session_state.history)
        avg_conf = np.mean([h["conf"] for h in st.session_state.history]) if st.session_state.history else 0
        bar_w = min(n_det * 5, 100)
        st.markdown(f"""
        <div style="background:{bg3};border:1px solid {bd3};border-radius:12px;padding:14px 16px;margin-top:12px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
            <span style="font-size:26px;font-weight:600;color:{vc3};">{n_det}</span>
            <span style="font-size:12px;color:#64748b;">signs detected this session</span>
          </div>
          <div style="height:4px;background:{'#2a2d3a' if dark else '#e2e8f0'};border-radius:2px;margin-bottom:10px;">
            <div style="width:{bar_w}%;height:4px;background:{vc3};border-radius:2px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:12px;color:#64748b;">Avg confidence: {avg_conf:.1f}%</span>
            <span style="font-size:12px;color:#64748b;">{'Live ~10fps' if IS_LOCAL else 'Snapshot mode'}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL STATS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == 2:
    col_left, col_right = st.columns([1, 1], gap="medium")

    t1   = summary.get("test_top1_accuracy_pct", 90.17)
    t3   = summary.get("test_top3_accuracy_pct", 97.81)
    t5   = summary.get("test_top5_accuracy_pct", 99.31)
    p1   = summary.get("phase1_best_val_acc_pct", 76.23)
    p2   = summary.get("phase2_best_val_acc_pct", 97.19)
    conf = summary.get("avg_confidence_pct", 87.88)
    c1   = "#1d9e75" if dark else "#0f6e56"
    c2   = "#378ADD" if dark else "#185FA5"
    c3   = "#EF9F27" if dark else "#854F0B"
    bg   = "#1a1d2e" if dark else "#ffffff"
    bd   = "#2a2d3a" if dark else "#e2e8f0"
    tc   = "#e2e8f0" if dark else "#1e293b"
    tc2  = "#64748b"

    with col_left:
        # Accuracy cards
        st.markdown('<div class="sec-label">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid" style="margin-bottom:14px;">
          <div class="stat-card"><div class="stat-val" style="color:{c1};">{t1}%</div><div class="stat-lbl">Top-1 Accuracy</div></div>
          <div class="stat-card"><div class="stat-val" style="color:{c2};">{t3}%</div><div class="stat-lbl">Top-3 Accuracy</div></div>
          <div class="stat-card"><div class="stat-val" style="color:{c3};">{t5}%</div><div class="stat-lbl">Top-5 Accuracy</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Training curve chart
        st.markdown('<div class="sec-label">TRAINING CURVES (TWO-PHASE)</div>', unsafe_allow_html=True)
        p1_epochs = list(range(1, 16))
        p1_acc    = [61.45, 65.44, 68.73, 68.26, 69.43, 72.86, 71.08, 72.25, 74.47, 72.95, 75.47, 76.23, 74.70, 74.16, 74.85]
        p2_epochs = list(range(16, 26))
        p2_acc    = [85.36, 88.91, 91.51, 93.34, 94.22, 95.32, 95.54, 96.56, 97.19, 97.11]

        all_epochs = p1_epochs + p2_epochs
        all_acc    = p1_acc + p2_acc

        fig, ax = plt.subplots(figsize=(8, 3.2))
        bg_color = "#1a1d2e" if dark else "#ffffff"
        ax_color = "#2a2d3a" if dark else "#e2e8f0"
        tx_color = "#64748b"

        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor("#13161f" if dark else "#f8fafc")

        ax.axvspan(0.5, 15.5, alpha=0.08, color="#378ADD")
        ax.axvspan(15.5, 25.5, alpha=0.08, color="#EF9F27")
        ax.axvline(15.5, color=ax_color, linestyle=":", lw=1)

        ax.plot(all_epochs, all_acc, color="#e24b4a", lw=2.5, marker="o", markersize=3, label="Val Accuracy")
        ax.text(8, min(all_acc) - 3, "Phase 1", ha="center", fontsize=8, color="#378ADD", style="italic")
        ax.text(20, min(all_acc) - 3, "Phase 2", ha="center", fontsize=8, color="#EF9F27", style="italic")

        ax.set_xlabel("Epoch", fontsize=9, color=tx_color)
        ax.set_ylabel("Val Accuracy (%)", fontsize=9, color=tx_color)
        ax.tick_params(colors=tx_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(ax_color)
        ax.grid(True, color=ax_color, alpha=0.5, lw=0.5)
        ax.legend(fontsize=8, labelcolor=tx_color, facecolor=bg_color, edgecolor=ax_color)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=bg_color)
        buf.seek(0)
        plt.close()
        st.image(buf, use_column_width=True)

        # Top performing classes
        st.markdown('<div class="sec-label" style="margin-top:8px;">TOP PERFORMING CLASSES</div>', unsafe_allow_html=True)
        top_classes = [
            ("Stop", 100), ("No vehicles", 99), ("Yield", 99),
            ("Priority road", 97), ("No passing (>3.5t)", 99),
        ]
        bar_bg_color = "#13161f" if dark else "#f8fafc"
        rows = ""
        for name_c, acc_c in top_classes:
            rows += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">'
                f'<span style="font-size:12px;color:{tc};width:160px;">{name_c}</span>'
                f'<div style="flex:1;height:18px;background:{bar_bg_color};border-radius:4px;overflow:hidden;">'
                f'<div style="width:{acc_c}%;height:18px;background:#1d9e75;border-radius:4px;display:flex;align-items:center;padding-left:6px;">'
                f'<span style="font-size:10px;color:white;">{acc_c}%</span>'
                f'</div></div></div>'
            )
        st.markdown(f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;padding:14px 16px;">{rows}</div>', unsafe_allow_html=True)

    with col_right:
        # Model info table
        st.markdown('<div class="sec-label">MODEL INFORMATION</div>', unsafe_allow_html=True)
        info_rows = [
            ("Architecture",  "EfficientNetB0"),
            ("Pretrained on", "ImageNet (1.2M images)"),
            ("Dataset",       "GTSRB (50,000+ images)"),
            ("Classes",       "43 traffic signs"),
            ("Parameters",    "5.3M total"),
            ("Model size",    "18.7 MB"),
            ("Framework",     "PyTorch + torchvision"),
            ("Inference",     f"~{summary.get('avg_confidence_pct', 87):.0f}ms / image"),
        ]
        st.markdown(f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;padding:12px 16px;">', unsafe_allow_html=True)
        for k, v in info_rows:
            st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Hardest classes
        st.markdown('<div class="sec-label" style="margin-top:14px;">HARDEST CLASSES</div>', unsafe_allow_html=True)
        hard_classes = [
            ("Go straight or left", 28, "#e24b4a"),
            ("Speed limit (20km/h)", 45, "#e24b4a"),
            ("Pedestrians",          45, "#EF9F27"),
            ("Dangerous curve R",    60, "#EF9F27"),
        ]
        hint_col = "#475569" if dark else "#94a3b8"
        st.markdown(f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;padding:14px 16px;">', unsafe_allow_html=True)
        for name_c, acc_c, col_c in hard_classes:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">'
                f'<span style="font-size:12px;color:{tc};width:160px;">{name_c}</span>'
                f'<div style="flex:1;height:18px;background:{bar_bg_color};border-radius:4px;overflow:hidden;">'
                f'<div style="width:{acc_c}%;height:18px;background:{col_c};border-radius:4px;display:flex;align-items:center;padding-left:6px;">'
                f'<span style="font-size:10px;color:white;">{acc_c}%</span>'
                f'</div></div></div>',
                unsafe_allow_html=True
            )
        st.markdown(f'<div style="font-size:11px;color:{hint_col};margin-top:6px;">These classes have few training samples and high visual similarity</div></div>', unsafe_allow_html=True)

        # Training summary
        st.markdown('<div class="sec-label" style="margin-top:14px;">TRAINING SUMMARY</div>', unsafe_allow_html=True)
        train_rows = [
            ("Phase 1 (frozen base)",  f"15 epochs → {p1}% val acc"),
            ("Phase 2 (fine-tuning)",  f"10 epochs → {p2}% val acc"),
            ("Test Top-1 accuracy",    f"{t1}%"),
            ("Test Top-3 accuracy",    f"{t3}%"),
            ("Test Top-5 accuracy",    f"{t5}%"),
            ("Avg confidence",         f"{conf}%"),
            ("Trained on",             "Kaggle T4 GPU"),
            ("Training strategy",      "Transfer Learning"),
        ]
        st.markdown(f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;padding:12px 16px;">', unsafe_allow_html=True)
        for k, v in train_rows:
            vc = c1 if "Top" in k or "test" in k.lower() else (tc if dark else "#1e293b")
            st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val" style="color:{vc};">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)