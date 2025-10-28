import streamlit as st
import cv2, numpy as np, yaml, os
from utils.io import read_bgr, align_to_ref
from utils.classical import detect_defects
from utils.viz import overlay_mask, draw_boxes
from utils.yolo_backend import YOLOBackend

st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔌 PCB Defect Detector")

with open('configs/default.yaml','r') as f:
    CFG = yaml.safe_load(f)

col1, col2 = st.columns(2)
ref = col1.file_uploader("Upload GOLDEN board (optional)", type=['png','jpg','jpeg'])
test = col2.file_uploader("Upload TEST board", type=['png','jpg','jpeg'])

backend = st.sidebar.selectbox("Backend", ["auto","yolo","classical"], index=0)

if test is not None:
    test_bytes = np.frombuffer(test.read(), np.uint8)
    test_bgr = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    ref_bgr = None
    if ref is not None:
        ref_bytes = np.frombuffer(ref.read(), np.uint8)
        ref_bgr = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
        ref_bgr = align_to_ref(ref_bgr, test_bgr)

    if backend == "yolo":
        try:
            y = YOLOBackend(CFG['inference']['model_path'])
            boxes, labels, scores = y.infer(test_bgr)
            vis = draw_boxes(test_bgr[..., ::-1], boxes) # RGB for display
            st.image(vis, use_column_width=True)
        except Exception as e:
            st.warning(f"YOLO unavailable ({e}). Falling back to classical.")
            m, boxes = detect_defects(test_bgr, ref_bgr, CFG['classical']['blur'], CFG['classical']['thr'], CFG['classical']['min_area'])
            st.image(overlay_mask(test_bgr[..., ::-1], m), use_column_width=True)
    else:
        m, boxes = detect_defects(test_bgr, ref_bgr, CFG['classical']['blur'], CFG['classical']['thr'], CFG['classical']['min_area'])
        st.subheader("Overlay")
        st.image(overlay_mask(test_bgr[..., ::-1], m), use_column_width=True)
        st.subheader("Boxes")
        st.image(draw_boxes(test_bgr[..., ::-1], boxes), use_column_width=True)
else:
    st.info("Upload at least a TEST board image.")
