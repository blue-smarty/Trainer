from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.artifacts import format_size, infer_onnx_path
from dashboard.backends.base import BackendAdapter, list_repo_paths
from dashboard.validation import (
    ValidationResult,
    validate_export_params,
    validate_setup_params,
    validate_train_params,
)


COMMON_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "custom / enter below",
]


class YOLODetectionBackend(BackendAdapter):
    key = "yolo_detection"
    label = "YOLO Detection"
    description = "Ultralytics YOLO object detection workflows."
    setup_supported = True

    def get_runs_root(self, repo_root: Path) -> Path | None:
        return repo_root / "runs" / "detect"

    def render_setup(self, repo_root: Path) -> None:
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

        if st.button("Run dataset setup", type="primary", key="setup_yolo"):
            result = validate_setup_params(dataset_root, classes)
            for msg in result.warnings:
                st.warning(msg)
            for msg in result.errors:
                st.error(msg)
            if result.ok:
                from scripts.setup_dataset import setup_dataset

                data_yaml_path = setup_dataset(root_path=dataset_root, classes_csv=classes)
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

    def render_train_controls(self, repo_root: Path) -> dict[str, Any]:
        data_yaml = st.selectbox(
            "Path to data.yaml",
            options=list_repo_paths(repo_root, "**/data.yaml", "data/my_dataset/data.yaml"),
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
                help="Enter model file name or local .pt path.",
            )
        else:
            model_name = model_choice

        col_ep, col_img, col_bat = st.columns(3)
        with col_ep:
            epochs = st.number_input("Epochs", min_value=1, value=50, key="tr_yolo_epochs")
        with col_img:
            imgsz = st.number_input(
                "Image size",
                min_value=32,
                value=640,
                step=32,
                key="tr_yolo_imgsz",
            )
        with col_bat:
            batch = st.number_input("Batch size", min_value=1, value=16, key="tr_yolo_batch")

        with st.expander("Advanced options"):
            col_proj, col_name = st.columns(2)
            with col_proj:
                project = st.text_input(
                    "Project directory",
                    value="runs/detect",
                    key="tr_yolo_project",
                )
            with col_name:
                run_name = st.text_input("Run name", value="train", key="tr_yolo_name")
            device = st.text_input("Device", value="", key="tr_yolo_device")
            cfg = st.text_input("Config yaml (optional)", value="", key="tr_yolo_cfg")
            resume = st.checkbox("Resume previous run", key="tr_yolo_resume")

        return {
            "data_yaml": data_yaml,
            "model_name": model_name,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "project": project,
            "name": run_name,
            "device": device.strip() or None,
            "cfg": cfg.strip() or None,
            "resume": resume,
        }

    def validate_train_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        return validate_train_params(
            data_yaml=config["data_yaml"],
            model_name=config["model_name"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            project=config["project"],
            repo_root=repo_root,
        )

    def train_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        from scripts.train import train_model

        data_path = (repo_root / config["data_yaml"]).resolve()
        project_root = Path(config["project"])
        if not project_root.is_absolute():
            project_root = (repo_root / project_root).resolve()

        before = self._get_run_candidates(project_root, config["name"])
        train_model(
            data=str(data_path),
            model_name=config["model_name"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            project=config["project"],
            name=config["name"],
            resume=config["resume"],
            device=config["device"],
            cfg=config["cfg"],
        )
        after = self._get_run_candidates(project_root, config["name"])
        run_dir = self._select_run_dir(after, before, project_root, config["name"])
        weights = sorted((run_dir / "weights").glob("*.pt")) if (run_dir / "weights").exists() else []
        return {"run_dir": run_dir, "weights": weights}

    def render_export_controls(self, repo_root: Path) -> dict[str, Any]:
        weights = st.selectbox(
            "Weights path",
            options=list_repo_paths(repo_root, "**/*.pt", "runs/detect/train/weights/best.pt"),
            key="weights_yolo",
            help="Select a trained .pt weights file to export.",
        )

        col_ei, col_eb = st.columns(2)
        with col_ei:
            imgsz = st.number_input(
                "Image size",
                min_value=32,
                value=640,
                step=32,
                key="ex_yolo_img",
            )
        with col_eb:
            batch = st.number_input("Batch size", min_value=1, value=1, key="ex_yolo_batch")

        with st.expander("Advanced options"):
            opset = st.number_input("ONNX opset", min_value=9, value=12, key="ex_yolo_opset")
            dynamic = st.checkbox("Dynamic shapes", key="ex_yolo_dynamic")

        return {
            "weights": weights,
            "imgsz": int(imgsz),
            "batch": int(batch),
            "opset": int(opset),
            "dynamic": dynamic,
        }

    def validate_export_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        return validate_export_params(
            weights=config["weights"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            opset=config["opset"],
            repo_root=repo_root,
        )

    def export_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        from scripts.export_hailo import export_onnx

        weights_path = (repo_root / config["weights"]).resolve()
        export_onnx(
            weights=str(weights_path),
            imgsz=config["imgsz"],
            batch=config["batch"],
            opset=config["opset"],
            dynamic=config["dynamic"],
        )
        return {"onnx_path": infer_onnx_path(weights_path)}

    @staticmethod
    def _get_run_candidates(project_root: Path, run_name_base: str) -> list[Path]:
        if not project_root.exists():
            return []
        candidates: list[Path] = []
        for p in project_root.iterdir():
            if p.is_dir() and (p / "weights").exists() and p.name.startswith(run_name_base):
                candidates.append(p)
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates

    @staticmethod
    def _select_run_dir(
        after: list[Path], before: list[Path], project_root: Path, requested_name: str
    ) -> Path:
        before_set = {p.resolve() for p in before}
        new_runs = [p for p in after if p.resolve() not in before_set]
        if new_runs:
            return new_runs[-1]
        if after:
            return after[-1]
        fallback = project_root / requested_name
        return fallback
