import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import io
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector using YOLOv8n",
    layout="wide"
)

# --- Standard Class Definitions ---
STANDARD_CLASSES = [
    'red blood cell', 
    'leukocyte', 
    'schizont', 
    'ring', 
    'gametocyte', 
    'trophozoite'
]
PARASITE_STAGES = ['schizont', 'ring', 'gametocyte', 'trophozoite']

# --- File Paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.onnx')
classes_path = os.path.join(base_path, 'classes.txt') 

# --- Validate Files ---
if not os.path.exists(model_path):
    st.error(f"ONNX model not found at: {model_path}")
else:
    st.success("✅ ONNX model found.")

if not os.path.exists(classes_path):
    st.warning(f"Class names file not found at: {classes_path}. Using hardcoded classes for validation.")
else:
    st.success(" ✅ Class names file found.")

# --- Load Model & Classes ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        net = cv2.dnn.readNet(model_path)
        return net
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}. Attempted path: {model_path}")
        return None

@st.cache_data
def load_class_names(classes_path):
    try:
        if os.path.exists(classes_path):
             with open(classes_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        else:
             return STANDARD_CLASSES 
    except Exception as e:
        st.error(f"Error loading class names from file: {e}. Using hardcoded standard list.")
        return STANDARD_CLASSES

net = load_onnx_model(model_path)
class_names = load_class_names(classes_path)

if len(class_names) != len(STANDARD_CLASSES):
    st.warning(f"Class count mismatch! File has {len(class_names)} classes, expected {len(STANDARD_CLASSES)}. Please verify 'classes.txt'.")
    class_names = STANDARD_CLASSES

st.title("🔬 Malaria Parasite (P.vivax) Detection using YOLOV8n")

# --- Dynamic Sidebar ---
st.sidebar.header("⚙️ Model & Visualization Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.35, 0.05)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_only_parasites = st.sidebar.checkbox("Show Only Parasite Detections", value=False)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Pastel"],
