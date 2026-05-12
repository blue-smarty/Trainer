#!/usr/bin/env python3
"""Streamlit dashboard for common Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.validation import (
    validate_setup_params,
    validate_train_params,
    validate_export_params,
)
from dashboard.artifacts import (
    find_recent_runs,
    find_all_onnx,
    format_size,
    format_mtime,
    infer_onnx_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMMON_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "custom / enter below",
]


def list_paths(pattern: str, default_value: str) -> list[str]:
    options: set[str] = {default_value}
    for path in REPO_ROOT.glob(pattern):
        if path.is_file():
            options.add(str(path.relative_to(REPO_ROOT)))
    return sorted(options)


def show_validation(result) -> bool:
    """Render validation errors and warnings; return True when safe to proceed."""
    for msg in result.warnings:
        st.warning(msg)
    for msg in result.errors:
        st.error(msg)
    return result.ok


def show_exception(exc: Exception) -> None:
    st.error(f"Operation failed: {exc}")
    with st.expander("Show traceback"):
        import traceback
        st.code(traceback.format_exc(), language="text")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Trainer Dashboard", layout="wide")
st.title("Trainer Dashboard")
st.caption("Run dataset setup, model training, and ONNX export from one place.")

# ---------------------------------------------------------------------------
# Sidebar — recent artifacts summary
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Recent Artifacts")
    _runs_root = REPO_ROOT / "runs" / "detect"
    _recent_runs = find_recent_runs(_runs_root, max_runs=5)
    if _recent_runs:
        for run in _recent_runs:
            with st.expander(f"📁 {run.name}", expanded=False):
                st.caption(f"Modified: {format_mtime(run.path)}")
                if run.best_pt:
                    st.markdown(f"✅ `best.pt` ({format_size(run.best_pt)})")
                if run.last_pt:
                    st.markdown(f"📄 `last.pt` ({format_size(run.last_pt)})")
                for onnx in run.onnx_files:
                    st.markdown(f"🔷 `{onnx.name}` ({format_size(onnx)})")
    else:
        st.info("No training runs found yet.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_setup, tab_train, tab_export, tab_artifacts = st.tabs(
    ["Setup Dataset", "Train Model", "Export ONNX", "Artifacts"]
)

# ── Setup Dataset ────────────────────────────────────────────────────────────

with tab_setup:
    st.subheader("Create YOLO dataset structure")
    st.markdown(
        "Creates the standard YOLOv8 directory layout and writes a `data.yaml` "
        "that you can use directly for training."
    )

    dataset_root = st.text_input(
        "Dataset root",
        value="data/my_dataset",
        help="Path (relative to repo root or absolute) where the dataset will be created.",
    )
    classes = st.text_input(
        "Classes (comma-separated)",
        value="person,car,bus",
        help="One or more class names separated by commas, e.g. `person,car,bus`.",
    )

    if st.button("Run dataset setup", type="primary"):
        result = validate_setup_params(dataset_root, classes)
        if show_validation(result):
            with st.spinner("Creating dataset structure…"):
                try:
                    from scripts.setup_dataset import setup_dataset

                    data_yaml_path = setup_dataset(
                        root_path=dataset_root, classes_csv=classes
                    )
                    st.success("Dataset structure created successfully.")
                    st.markdown("**Generated file**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(str(data_yaml_path), language="text")
                    with col2:
                        st.metric("data.yaml", format_size(data_yaml_path))

                    with st.expander("Preview data.yaml"):
                        with open(data_yaml_path, "r", encoding="utf-8") as fh:
                            st.code(fh.read(), language="yaml")
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

# ── Train Model ──────────────────────────────────────────────────────────────

with tab_train:
    st.subheader("Train YOLOv8 model")

    data_yaml = st.selectbox(
        "Path to data.yaml",
        options=list_paths("**/data.yaml", "data/my_dataset/data.yaml"),
        help="Select the data.yaml that describes your dataset.",
    )

    model_choice = st.selectbox(
        "Model",
        options=COMMON_MODELS,
        index=0,
        help="Choose a YOLOv8 model size. Larger models are more accurate but slower.",
    )
    if model_choice == "custom / enter below":
        model_name = st.text_input(
            "Custom model name or path",
            value="yolov8n.pt",
            help="Enter the model file name (downloaded automatically) or a path to a local .pt file.",
        )
    else:
        model_name = model_choice

    col_ep, col_img, col_bat = st.columns(3)
    with col_ep:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            value=50,
            help="Number of full passes through the training data.",
        )
    with col_img:
        imgsz = st.number_input(
            "Image size",
            min_value=32,
            value=640,
            step=32,
            help="Input resolution (pixels). Must be a multiple of 32.",
        )
    with col_bat:
        batch = st.number_input(
            "Batch size",
            min_value=1,
            value=16,
            help="Number of images processed per training step.",
        )

    with st.expander("Advanced options"):
        col_proj, col_name = st.columns(2)
        with col_proj:
            project = st.text_input(
                "Project directory",
                value="runs/detect",
                help="Parent folder where run output is saved.",
            )
        with col_name:
            run_name = st.text_input(
                "Run name",
                value="train",
                help="Sub-folder name inside the project directory for this run.",
            )
        device = st.text_input(
            "Device",
            value="",
            help=(
                "Training device: leave blank for auto-detect, `cpu` for CPU, "
                "`0` for first GPU, `0,1` for multi-GPU, or RTX 2060 aliases "
                "(`rtx2060`, `geforce rtx 2060`, `nvidia geforce rtx 2060`)."
            ),
        )
        cfg = st.text_input(
            "Config yaml (optional)",
            value="",
            help="Path to an Ultralytics trainer config file (e.g. `configs/yolov8-default.yaml`).",
        )
        resume = st.checkbox(
            "Resume previous run",
            help="Continue training from the last saved checkpoint of a previous run.",
        )

    if st.button("Run training", type="primary"):
        result = validate_train_params(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            project=project,
            repo_root=REPO_ROOT,
        )
        if show_validation(result):
            data_path = (REPO_ROOT / data_yaml).resolve()
            with st.spinner("Training in progress — this may take a while…"):
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
                    st.success("Training completed successfully.")

                    run_dir = REPO_ROOT / project / run_name
                    weights_dir = run_dir / "weights"
                    st.markdown("**Training output**")
                    st.code(str(run_dir), language="text")

                    if weights_dir.exists():
                        found_weights = sorted(weights_dir.glob("*.pt"))
                        if found_weights:
                            st.markdown("**Weights found:**")
                            for wt in found_weights:
                                st.markdown(f"- `{wt.relative_to(REPO_ROOT)}` ({format_size(wt)})")
                    else:
                        st.info("Weights directory not found yet; check the run directory above.")
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

# ── Export ONNX ───────────────────────────────────────────────────────────────

with tab_export:
    st.subheader("Export trained model to ONNX")
    st.markdown(
        "Export a trained `.pt` weights file to ONNX format, ready for the "
        "Hailo Dataflow Compiler."
    )

    weights = st.selectbox(
        "Weights path",
        options=list_paths("**/*.pt", "runs/detect/train/weights/best.pt"),
        key="weights",
        help="Select a trained .pt weights file to export.",
    )

    col_ei, col_eb = st.columns(2)
    with col_ei:
        export_imgsz = st.number_input(
            "Image size",
            min_value=32,
            value=640,
            step=32,
            key="ex_img",
            help="Input resolution to bake into the ONNX graph.",
        )
    with col_eb:
        export_batch = st.number_input(
            "Batch size",
            min_value=1,
            value=1,
            key="ex_batch",
            help="Batch dimension in the ONNX model (usually 1 for Hailo).",
        )

    with st.expander("Advanced options"):
        opset = st.number_input(
            "ONNX opset",
            min_value=9,
            value=12,
            help="ONNX operator-set version. Hailo recommends opset 11 or 12.",
        )
        dynamic = st.checkbox(
            "Dynamic shapes",
            help="Enable variable batch/spatial dimensions in the exported ONNX.",
        )

    if st.button("Run ONNX export", type="primary"):
        result = validate_export_params(
            weights=weights,
            imgsz=int(export_imgsz),
            batch=int(export_batch),
            opset=int(opset),
            repo_root=REPO_ROOT,
        )
        if show_validation(result):
            weights_path = (REPO_ROOT / weights).resolve()
            with st.spinner("Exporting to ONNX…"):
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

                    onnx_path = infer_onnx_path(weights_path)
                    if onnx_path:
                        st.markdown("**Exported file**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.code(str(onnx_path), language="text")
                        with col2:
                            st.metric("ONNX size", format_size(onnx_path))
                    else:
                        st.info(
                            "ONNX file not found next to weights. "
                            "Check the weights directory for a .onnx file."
                        )
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

# ── Artifacts ─────────────────────────────────────────────────────────────────

with tab_artifacts:
    st.subheader("Artifacts Browser")
    st.markdown(
        "Browse recent training runs, weights, and exported ONNX files without "
        "leaving the dashboard."
    )

    runs_root = REPO_ROOT / "runs" / "detect"
    recent_runs = find_recent_runs(runs_root, max_runs=20)

    if not recent_runs:
        st.info(
            f"No training runs found under `{runs_root.relative_to(REPO_ROOT)}`. "
            "Complete a training run first."
        )
    else:
        st.markdown(f"**{len(recent_runs)} run(s) found** — sorted newest first")
        for run in recent_runs:
            with st.expander(f"📁 {run.name}  —  {format_mtime(run.path)}", expanded=False):
                st.markdown(f"**Path:** `{run.path}`")

                if run.weights:
                    st.markdown("**Weights:**")
                    for wt in run.weights:
                        label = "✅ best.pt" if wt.name == "best.pt" else f"📄 {wt.name}"
                        st.markdown(
                            f"- {label} &nbsp; `{wt}` &nbsp; ({format_size(wt)}, "
                            f"modified {format_mtime(wt)})"
                        )
                else:
                    st.caption("No .pt weights found in this run.")

                if run.onnx_files:
                    st.markdown("**ONNX exports:**")
                    for onnx in run.onnx_files:
                        st.markdown(
                            f"- 🔷 `{onnx}` ({format_size(onnx)}, "
                            f"modified {format_mtime(onnx)})"
                        )

    st.divider()
    st.markdown("#### All ONNX files in repository")
    all_onnx = find_all_onnx(REPO_ROOT)
    if all_onnx:
        for onnx in all_onnx:
            st.markdown(
                f"- 🔷 `{onnx}` ({format_size(onnx)}, modified {format_mtime(onnx)})"
            )
    else:
        st.info("No .onnx files found in the repository yet.")
