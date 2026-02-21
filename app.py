import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Object Detection AI", page_icon="🔍", layout="wide")

st.title("🔍 Custom Object Detection")
st.write("This model was trained from scratch on the PASCAL VOC 2012 dataset.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    # This loads the YOLO model and caches it so it doesn't reload on every interaction
    return YOLO(model_path)

# Path to your downloaded weights
MODEL_FILE = "best.pt"

if not os.path.exists(MODEL_FILE):
    st.error(f"**Error:** '{MODEL_FILE}' not found!")
    st.info("Please download 'best.pt' from your Colab 'runs' directory and place it in the same folder as this script.")
else:
    model = load_model(MODEL_FILE)
    st.success("Model loaded successfully!")

    # --- SIDEBAR ---
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, help="Lower values show more (but less certain) detections.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Project Details
    - **Architecture:** YOLOv11n
    - **Dataset:** PASCAL VOC 2012
    - **Classes:** 20 (Person, Car, Dog, etc.)
    - **Training:** 20 Epochs (From Scratch)
    """)

    # --- UPLOAD SECTION ---
    uploaded_file = st.file_uploader("Upload an image for detection...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Run Prediction
        with st.spinner('Running AI inference...'):
            # Convert PIL image to numpy array for YOLO
            results = model.predict(source=image, conf=confidence)
            
            # Plot results on the image (returns BGR)
            res_plotted = results[0].plot()
            
            # Convert BGR (OpenCV) back to RGB (Streamlit/PIL)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Display Results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.header("AI Detections")
            st.image(res_plotted_rgb, use_container_width=True)

        # Show detection data table
        if len(results[0].boxes) > 0:
            st.subheader("Detections Found:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"✅ **{label.capitalize()}** — {conf:.2%}")
        else:
            st.warning("No objects detected. Try lowering the confidence threshold in the sidebar.")

st.divider()
st.caption("Developed as a third Year Computer Vision Project.")
