#!/usr/bin/env python3
"""Streamlit dashboard for common Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.setup_dataset import setup_dataset


def show_exception(exc: Exception) -> None:
    st.error(f"Operation failed: {exc}")


st.set_page_config(page_title="Trainer Dashboard", layout="wide")
st.title("Trainer Dashboard")
st.caption("Run dataset setup, model training, and ONNX export from one place.")

tab_setup, tab_train, tab_export = st.tabs(
    ["Setup Dataset", "Train Model", "Export ONNX"]
)

with tab_setup:
    st.subheader("Create YOLO dataset structure")
    dataset_root = st.text_input("Dataset root", value="data/my_dataset")
    classes = st.text_input("Classes (comma-separated)", value="person,car,bus")
    if st.button("Run dataset setup"):
        try:
            data_yaml_path = setup_dataset(root_path=dataset_root, classes_csv=classes)
            st.success(f"Created dataset structure and wrote: {data_yaml_path}")
        except Exception as exc:  # pragma: no cover - UI feedback path
            show_exception(exc)

with tab_train:
    st.subheader("Train YOLOv8 model")
    data_yaml = st.text_input("Path to data.yaml", value="data/my_dataset/data.yaml")
    model_name = st.text_input("Model", value="yolov8n.pt")
    epochs = st.number_input("Epochs", min_value=1, value=50)
    imgsz = st.number_input("Image size", min_value=32, value=640)
    batch = st.number_input("Batch size", min_value=1, value=16)
    device = st.text_input("Device (optional)", value="")
    project = st.text_input("Project directory", value="runs/detect")
    run_name = st.text_input("Run name", value="train")
    cfg = st.text_input("Config yaml (optional)", value="")
    resume = st.checkbox("Resume previous run")
    if st.button("Run training"):
        data_path = Path(data_yaml).expanduser()
        if not data_path.exists():
            st.error(f"data.yaml not found: {data_path}")
        else:
            try:
                from scripts.train import train_model

                train_model(
                    data=str(data_path),
                    model_name=model_name,
                    epochs=int(epochs),
                    imgsz=int(imgsz),
                    batch=int(batch),
                    project=project,
                    name=run_name,
                    resume=resume,
                    device=device.strip() or None,
                    cfg=cfg.strip() or None,
                )
                st.success("Training command started and completed.")
            except Exception as exc:  # pragma: no cover - UI feedback path
                show_exception(exc)

with tab_export:
    st.subheader("Export trained model to ONNX")
    weights = st.text_input(
        "Weights path", value="runs/detect/train/weights/best.pt", key="weights"
    )
    export_imgsz = st.number_input("Image size", min_value=32, value=640, key="ex_img")
    export_batch = st.number_input("Batch size", min_value=1, value=1, key="ex_batch")
    opset = st.number_input("ONNX opset", min_value=9, value=12)
    dynamic = st.checkbox("Dynamic shapes")
    if st.button("Run ONNX export"):
        weights_path = Path(weights).expanduser()
        if not weights_path.exists():
            st.error(f"Weights file not found: {weights_path}")
        else:
            try:
                from scripts.export_hailo import export_onnx

                export_onnx(
                    weights=str(weights_path),
                    imgsz=int(export_imgsz),
                    batch=int(export_batch),
                    opset=int(opset),
                    dynamic=dynamic,
                )
                st.success("ONNX export completed.")
            except Exception as exc:  # pragma: no cover - UI feedback path
                show_exception(exc)
