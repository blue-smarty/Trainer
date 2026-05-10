#!/usr/bin/env python3
"""Streamlit dashboard for common Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run_command(command: list[str]) -> None:
    """Run a repository script and show output."""
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    st.code(" ".join(command), language="bash")
    if result.stdout:
        st.text_area("stdout", value=result.stdout, height=180)
    if result.stderr:
        st.text_area("stderr", value=result.stderr, height=180)
    if result.returncode == 0:
        st.success("Command completed successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")


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
        run_command(
            [
                sys.executable,
                str(SCRIPTS_DIR / "setup_dataset.py"),
                "--root",
                dataset_root,
                "--classes",
                classes,
            ]
        )

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
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "train.py"),
            "--data",
            data_yaml,
            "--model",
            model_name,
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--project",
            project,
            "--name",
            run_name,
        ]
        if device.strip():
            cmd.extend(["--device", device.strip()])
        if cfg.strip():
            cmd.extend(["--cfg", cfg.strip()])
        if resume:
            cmd.append("--resume")
        run_command(cmd)

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
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "export_hailo.py"),
            "--weights",
            weights,
            "--imgsz",
            str(export_imgsz),
            "--batch",
            str(export_batch),
            "--opset",
            str(opset),
        ]
        if dynamic:
            cmd.append("--dynamic")
        run_command(cmd)
