import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Sprite Detector", page_icon="🥤")

st.markdown(
    """
    <h1 style='text-align: left; margin-bottom: 30px;'>
        Sprite Detection Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)
# Cache model so it loads only once
@st.cache_resource
def load_model():
    # return YOLO("runs/detect/sprite_model_v1/weights/best.pt")
    return YOLO("best.pt")

model_sprite = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    
    image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    results = model_sprite(image_bgr, conf=0.6)

    # --- Accurate sprite count ---
    sprite_count = sum(
        1 for box in results[0].boxes
        if model_sprite.names[int(box.cls[0])] == "sprite"
    )
    
    # --- Draw bounding boxes ONLY for sprite ---
    annotated = image_rgb.copy()

    for box in results[0].boxes:
        if model_sprite.names[int(box.cls[0])] == "sprite":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = f"Sprite {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                 0.8,
                (0, 255, 0),
               2,
            )

    # --- Show results ---
    st.image(annotated, caption="Sprite Detections", width="stretch")
    st.markdown(
    f"<div style='font-size:28px; font-weight:600;'>"
    f"Sprite bottles detected: {sprite_count}"
    f"</div>",
    unsafe_allow_html=True
)