import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import imageio
import platform

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Crowd AI", layout="wide")

st.markdown("""
<h1 style='text-align: center;'>
🚨 Predictive Crowd AI for Stampede Prevention
</h1>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider("Alert Threshold", 1, 100, 20)

mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["Upload Video"]
)

# (Disable camera on cloud)
if platform.system() != "Windows":
    mode = "Upload Video"

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# ---------- LOAD MODEL ----------
model = YOLO("yolov8n.pt")  # lightweight model (faster on cloud)

# ---------- HEATMAP (NO OPENCV) ----------
def generate_heatmap(frame, boxes):
    h, w, _ = frame.shape
    heatmap = np.zeros((h, w), dtype=float)

    for (x1, y1, x2, y2) in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        x_min = max(cx - 15, 0)
        x_max = min(cx + 15, w)
        y_min = max(cy - 15, 0)
        y_max = min(cy + 15, h)

        heatmap[y_min:y_max, x_min:x_max] += 1

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = (heatmap * 255).astype(np.uint8)

    # Create red heat overlay
    colored = np.zeros_like(frame)
    colored[:, :, 0] = heatmap  # Red channel

    return np.clip(frame + colored, 0, 255)

# ---------- LAYOUT ----------
col1, col2 = st.columns([3, 1])

with col1:
    FRAME_WINDOW = st.image([])

with col2:
    st.markdown("## 📊 Live Stats")
    people_placeholder = st.empty()
    density_placeholder = st.empty()
    risk_placeholder = st.empty()
    prediction_placeholder = st.empty()
    alert_placeholder = st.empty()

# ---------- VIDEO PROCESS ----------
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    reader = imageio.get_reader(tfile.name)

    for frame in reader:
        frame = np.array(frame)

        results = model(frame, conf=0.3)

        count = 0
        boxes = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person class
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

                    # Draw box (manual using numpy)
                    frame[y1:y1+3, x1:x2] = [0, 255, 0]
                    frame[y2-3:y2, x1:x2] = [0, 255, 0]
                    frame[y1:y2, x1:x1+3] = [0, 255, 0]
                    frame[y1:y2, x2-3:x2] = [0, 255, 0]

        # Apply heatmap
        frame = generate_heatmap(frame, boxes)

        # ---------- CALCULATIONS ----------
        density = count / 50

        if count < threshold:
            risk = "SAFE"
            prediction = "STABLE"
            alert = "✅ SAFE"
        else:
            risk = "HIGH"
            prediction = "INCREASING RISK"
            alert = "🚨 DANGER"

        # ---------- UPDATE UI ----------
        people_placeholder.markdown(f"👥 **People Count:** {count}")
        density_placeholder.markdown(f"📏 **Density:** {density:.2f}")
        risk_placeholder.markdown(f"⚠️ **Risk Level:** {risk}")
        prediction_placeholder.markdown(f"🔮 **Prediction:** {prediction}")
        alert_placeholder.markdown(f"🚨 **Alert:** {alert}")

        FRAME_WINDOW.image(frame)
