import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sprite Stock Intelligence",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0c10 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(0,230,118,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(0,188,212,0.05) 0%, transparent 60%),
        #0a0c10 !important;
}

[data-testid="stHeader"], footer { display: none !important; }

/* ── Main container ── */
.block-container {
    max-width: 1100px !important;
    padding: 2.5rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.hero-header {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 0.4rem;
}
.hero-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #00e676, #00bcd4);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 24px rgba(0,230,118,0.35);
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 40%, #00e676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.88rem;
    color: #6b7280;
    margin: 0.2rem 0 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(0,230,118,0.4), rgba(0,188,212,0.2), transparent);
    margin: 1.5rem 0 2rem;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00e676;
    margin-bottom: 0.6rem;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(0,230,118,0.25) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,230,118,0.5) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] label {
    color: #9ca3af !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(0,230,118,0.12) !important;
    border: 1px solid rgba(0,230,118,0.3) !important;
    color: #00e676 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Metrics row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 2rem 0 1.5rem;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent, #00e676), transparent);
}
.metric-card:hover {
    border-color: rgba(0,230,118,0.2);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    color: #ffffff;
}
.metric-sub {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.3rem;
}

/* ── Status banner ── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1rem 1.4rem;
    border-radius: 14px;
    border: 1px solid;
    margin-bottom: 1.6rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
}
.status-ok {
    background: rgba(0,230,118,0.07);
    border-color: rgba(0,230,118,0.3);
    color: #00e676;
}
.status-warn {
    background: rgba(255,179,0,0.07);
    border-color: rgba(255,179,0,0.35);
    color: #ffb300;
}
.status-crit {
    background: rgba(255,82,82,0.08);
    border-color: rgba(255,82,82,0.4);
    color: #ff5252;
}
.status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    background: currentColor;
    box-shadow: 0 0 10px currentColor;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}
.status-badge {
    margin-left: auto;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid currentColor;
    opacity: 0.8;
}

/* ── Image display ── */
.image-section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.image-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}
[data-testid="stImage"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Confidence threshold note ── */
.conf-note {
    font-size: 0.78rem;
    color: #4b5563;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] p {
    color: #00e676 !important;
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🥤</div>
    <div>
        <p class="hero-title">Stock Intelligence</p>
        <p class="hero-sub">Automated Sprite bottle detection & inventory monitoring</p>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model_sprite = load_model()


# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag & drop a shelf image or click to browse",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed",
)
st.markdown('<div class="conf-note">⚙ Detection confidence threshold: 60% &nbsp;|&nbsp; Supported formats: JPG, PNG, JPEG</div>', unsafe_allow_html=True)


# ── Detection ──────────────────────────────────────────────────────────────────
if uploaded_file:
    image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing image…"):
        t0 = time.time()
        results = model_sprite(image_bgr, conf=0.6)
        elapsed = time.time() - t0

    boxes = results[0].boxes
    sprite_count = sum(
        1 for box in boxes
        if model_sprite.names[int(box.cls[0])] == "sprite"
    )
    total_objects = len(boxes)
    avg_conf = (
        float(np.mean([float(b.conf[0]) for b in boxes])) * 100
        if total_objects > 0 else 0
    )

    # ── Status logic ──────────────────────────────────────────────────────────
    if sprite_count > 10:
        status_cls, status_icon, status_text, badge = "status-ok",   "✓", "Stock levels are healthy",            "OPTIMAL"
    elif sprite_count > 2:
        status_cls, status_icon, status_text, badge = "status-warn", "⚠", "Stock is below recommended threshold", "LOW STOCK"
    else:
        status_cls, status_icon, status_text, badge = "status-crit", "✕", "Critical — immediate restocking required", "CRITICAL"

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card" style="--accent:#00e676;">
            <div class="metric-label">Sprite Bottles</div>
            <div class="metric-value">{sprite_count}</div>
            <div class="metric-sub">detected in frame</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="status-banner {status_cls}">
        <div class="status-dot"></div>
        <span>{status_icon}&nbsp; {status_text}</span>
        <span class="status-badge">{badge}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="image-section-label">Shelf Image</div>', unsafe_allow_html=True)
    st.image(image_rgb, use_container_width=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align:center;
        padding: 4rem 2rem;
        background: rgba(255,255,255,0.015);
        border: 1.5px dashed rgba(255,255,255,0.07);
        border-radius: 20px;
        margin-top: 2rem;
    ">
        <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">📷</div>
        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.1rem; color:#374151; margin-bottom:0.4rem;">No image uploaded yet</div>
        <div style="font-size:0.85rem; color:#374151;">Upload a shelf image above to begin detection</div>
    </div>
    """, unsafe_allow_html=True)