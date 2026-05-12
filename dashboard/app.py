#!/usr/bin/env python3
"""Streamlit dashboard for common Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.artifacts import find_all_onnx, find_recent_runs, format_mtime, format_size
from dashboard.backends import get_backend_map


def show_validation(result) -> bool:
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


def show_train_result(payload: dict[str, Any]) -> None:
    run_dir = payload.get("run_dir")
    if run_dir:
        st.markdown("**Training output**")
        st.code(str(run_dir), language="text")
    weights = payload.get("weights") or []
    if weights:
        st.markdown("**Weights found:**")
        for wt in weights:
            st.markdown(f"- `{wt}` ({format_size(wt)})")
    if "output" in payload:
        st.markdown("**Backend output**")
        st.code(repr(payload["output"]), language="text")


def show_export_result(payload: dict[str, Any]) -> None:
    onnx_path = payload.get("onnx_path")
    if onnx_path:
        st.markdown("**Exported file**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.code(str(onnx_path), language="text")
        with col2:
            st.metric("ONNX size", format_size(onnx_path))
    if "output" in payload:
        st.markdown("**Backend output**")
        st.code(repr(payload["output"]), language="text")


st.set_page_config(page_title="Trainer Dashboard", layout="wide")
st.title("Trainer Dashboard")
st.caption("Run dataset setup, model training, and ONNX export from one place.")

backend_map = get_backend_map()
backend_options = list(backend_map.keys())

with st.sidebar:
    st.header("Task / Backend")
    active_backend_key = st.selectbox(
        "Choose workflow",
        options=backend_options,
        format_func=lambda key: backend_map[key].label,
    )
    active_backend = backend_map[active_backend_key]
    st.caption(active_backend.description)
    st.success(f"Active backend: {active_backend.label}")

    st.divider()
    st.header("Recent Artifacts")
    runs_root = active_backend.get_runs_root(REPO_ROOT)
    if runs_root:
        recent_runs = find_recent_runs(runs_root, max_runs=5)
        if recent_runs:
            for run in recent_runs:
                with st.expander(f"📁 {run.name}", expanded=False):
                    st.caption(f"Modified: {format_mtime(run.path)}")
                    if run.best_pt:
                        st.markdown(f"✅ `best.pt` ({format_size(run.best_pt)})")
                    if run.last_pt:
                        st.markdown(f"📄 `last.pt` ({format_size(run.last_pt)})")
        else:
            st.info("No training runs found yet.")
    else:
        st.info("Recent run discovery is backend-defined.")


tab_setup, tab_train, tab_export, tab_artifacts = st.tabs(
    ["Setup Dataset", "Train Model", "Export ONNX", "Artifacts"]
)

with tab_setup:
    if active_backend.setup_supported:
        try:
            active_backend.render_setup(REPO_ROOT)
        except Exception as exc:
            show_exception(exc)
    else:
        st.info(f"{active_backend.label} does not require dashboard-managed setup.")

with tab_train:
    st.subheader(f"Train — {active_backend.label}")
    train_config = active_backend.render_train_controls(REPO_ROOT)
    if st.button(f"Run training ({active_backend.label})", type="primary"):
        result = active_backend.validate_train_config(train_config, REPO_ROOT)
        if show_validation(result):
            with st.spinner("Training in progress — this may take a while…"):
                try:
                    payload = active_backend.train_model(train_config, REPO_ROOT)
                    st.success("Training completed successfully.")
                    show_train_result(payload)
                except Exception as exc:
                    show_exception(exc)

with tab_export:
    st.subheader(f"Export — {active_backend.label}")
    export_config = active_backend.render_export_controls(REPO_ROOT)
    if st.button(f"Run export ({active_backend.label})", type="primary"):
        result = active_backend.validate_export_config(export_config, REPO_ROOT)
        if show_validation(result):
            with st.spinner("Exporting…"):
                try:
                    payload = active_backend.export_model(export_config, REPO_ROOT)
                    st.success("Export completed.")
                    show_export_result(payload)
                except Exception as exc:
                    show_exception(exc)

with tab_artifacts:
    st.subheader(f"Artifacts Browser — {active_backend.label}")
    runs_root = active_backend.get_runs_root(REPO_ROOT)
    if runs_root:
        recent_runs = find_recent_runs(runs_root, max_runs=20)
        if not recent_runs:
            st.info(f"No runs found under `{runs_root}`.")
        else:
            st.markdown(f"**{len(recent_runs)} run(s) found** — sorted newest first")
            for run in recent_runs:
                with st.expander(f"📁 {run.name}  —  {format_mtime(run.path)}", expanded=False):
                    st.markdown(f"**Path:** `{run.path}`")
                    if run.weights:
                        st.markdown("**Weights:**")
                        for wt in run.weights:
                            st.markdown(f"- `{wt}` ({format_size(wt)}, modified {format_mtime(wt)})")
                    else:
                        st.caption("No .pt weights found in this run.")
                    if run.onnx_files:
                        st.markdown("**ONNX exports:**")
                        for onnx in run.onnx_files:
                            st.markdown(
                                f"- `{onnx}` ({format_size(onnx)}, modified {format_mtime(onnx)})"
                            )
    else:
        st.info("Run artifacts are managed by your custom backend scripts.")

    st.divider()
    st.markdown("#### All ONNX files in repository")
    all_onnx = find_all_onnx(REPO_ROOT)
    if all_onnx:
        for onnx in all_onnx:
            st.markdown(f"- `{onnx}` ({format_size(onnx)}, modified {format_mtime(onnx)})")
    else:
        st.info("No .onnx files found in the repository yet.")
