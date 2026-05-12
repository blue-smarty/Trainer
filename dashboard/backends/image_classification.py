from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.artifacts import infer_onnx_path
from dashboard.backends.base import BackendAdapter, list_repo_paths
from dashboard.validation import ValidationResult


CLASSIFICATION_MODELS = [
    "yolov8n-cls.pt",
    "yolov8s-cls.pt",
    "yolov8m-cls.pt",
    "custom / enter below",
]


class ImageClassificationBackend(BackendAdapter):
    key = "image_classification"
    label = "Image Classification"
    description = "Ultralytics image classification training/export workflows."

    def get_runs_root(self, repo_root: Path) -> Path | None:
        return repo_root / "runs" / "classify"

    def render_train_controls(self, repo_root: Path) -> dict[str, Any]:
        dataset = st.text_input(
            "Dataset directory",
            value="data/classification",
            help="Folder with class subfolders in `train/` and `val/`.",
            key="tr_cls_dataset",
        )
        model_choice = st.selectbox(
            "Model",
            options=CLASSIFICATION_MODELS,
            index=0,
            key="tr_cls_model_choice",
        )
        if model_choice == "custom / enter below":
            model_name = st.text_input(
                "Custom model name or path",
                value="yolov8n-cls.pt",
                key="tr_cls_model_name",
            )
        else:
            model_name = model_choice

        col_ep, col_img, col_bat = st.columns(3)
        with col_ep:
            epochs = st.number_input("Epochs", min_value=1, value=30, key="tr_cls_epochs")
        with col_img:
            imgsz = st.number_input(
                "Image size", min_value=32, value=224, step=32, key="tr_cls_imgsz"
            )
        with col_bat:
            batch = st.number_input("Batch size", min_value=1, value=32, key="tr_cls_batch")

        with st.expander("Advanced options"):
            col_proj, col_name = st.columns(2)
            with col_proj:
                project = st.text_input(
                    "Project directory",
                    value="runs/classify",
                    key="tr_cls_project",
                )
            with col_name:
                run_name = st.text_input("Run name", value="train", key="tr_cls_name")
            device = st.text_input("Device", value="", key="tr_cls_device")
            resume = st.checkbox("Resume previous run", key="tr_cls_resume")

        return {
            "dataset": dataset,
            "model_name": model_name,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "project": project,
            "name": run_name,
            "device": device.strip() or None,
            "resume": resume,
        }

    def validate_train_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        dataset_path = Path(config["dataset"])
        if not dataset_path.is_absolute():
            dataset_path = (repo_root / dataset_path).resolve()
        if not dataset_path.exists():
            errors.append(f"Dataset directory not found: {dataset_path}")
        else:
            for split in ("train", "val"):
                if not (dataset_path / split).exists():
                    warnings.append(f"Expected split folder missing: {(dataset_path / split)}")
        if not config["model_name"]:
            errors.append("Model name must not be empty.")
        if config["epochs"] < 1:
            errors.append("Epochs must be at least 1.")
        if config["imgsz"] < 32:
            errors.append("Image size must be at least 32.")
        if config["batch"] < 1:
            errors.append("Batch size must be at least 1.")
        if not config["project"]:
            errors.append("Project directory must not be empty.")
        return ValidationResult(errors=errors, warnings=warnings)

    def train_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        from ultralytics import YOLO

        dataset_path = Path(config["dataset"])
        if not dataset_path.is_absolute():
            dataset_path = (repo_root / dataset_path).resolve()
        model = YOLO(config["model_name"])
        train_kwargs = {
            "task": "classify",
            "data": str(dataset_path),
            "epochs": config["epochs"],
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "project": config["project"],
            "name": config["name"],
            "resume": config["resume"],
        }
        if config["device"] is not None:
            train_kwargs["device"] = config["device"]
        model.train(**train_kwargs)
        run_dir = self._resolve_run_dir(repo_root, config["project"], config["name"])
        weights = sorted((run_dir / "weights").glob("*.pt")) if (run_dir / "weights").exists() else []
        return {"run_dir": run_dir, "weights": weights}

    def render_export_controls(self, repo_root: Path) -> dict[str, Any]:
        weights = st.selectbox(
            "Weights path",
            options=list_repo_paths(repo_root, "**/*.pt", "runs/classify/train/weights/best.pt"),
            key="weights_cls",
        )
        col_ei, col_eb = st.columns(2)
        with col_ei:
            imgsz = st.number_input(
                "Image size", min_value=32, value=224, step=32, key="ex_cls_imgsz"
            )
        with col_eb:
            batch = st.number_input("Batch size", min_value=1, value=1, key="ex_cls_batch")
        with st.expander("Advanced options"):
            opset = st.number_input("ONNX opset", min_value=9, value=12, key="ex_cls_opset")
            dynamic = st.checkbox("Dynamic shapes", key="ex_cls_dynamic")
        return {
            "weights": weights,
            "imgsz": int(imgsz),
            "batch": int(batch),
            "opset": int(opset),
            "dynamic": dynamic,
        }

    def validate_export_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        weights_path = Path(config["weights"])
        if not weights_path.is_absolute():
            weights_path = (repo_root / weights_path).resolve()
        if not weights_path.exists():
            errors.append(f"Weights file not found: {weights_path}")
        elif weights_path.suffix.lower() != ".pt":
            warnings.append(f"Weights file does not end with .pt: {weights_path.name}")
        if config["imgsz"] < 32:
            errors.append("Image size must be at least 32.")
        if config["batch"] < 1:
            errors.append("Batch size must be at least 1.")
        if config["opset"] < 9:
            errors.append("ONNX opset version must be at least 9.")
        return ValidationResult(errors=errors, warnings=warnings)

    def export_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        from ultralytics import YOLO

        weights_path = Path(config["weights"])
        if not weights_path.is_absolute():
            weights_path = (repo_root / weights_path).resolve()
        model = YOLO(str(weights_path))
        model.export(
            format="onnx",
            imgsz=config["imgsz"],
            batch=config["batch"],
            opset=config["opset"],
            dynamic=config["dynamic"],
            simplify=True,
        )
        return {"onnx_path": infer_onnx_path(weights_path)}

    @staticmethod
    def _resolve_run_dir(repo_root: Path, project: str, name: str) -> Path:
        project_root = Path(project)
        if not project_root.is_absolute():
            project_root = (repo_root / project_root).resolve()
        if not project_root.exists():
            return project_root / name
        candidates = [p for p in project_root.iterdir() if p.is_dir() and p.name.startswith(name)]
        if not candidates:
            return project_root / name
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]
