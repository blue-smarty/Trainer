#!/usr/bin/env python3
"""FastAPI web UI for common Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Annotated, Any

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_hailo import export_onnx
from scripts.setup_dataset import setup_dataset
from scripts.train import train_model


app = FastAPI(title="Trainer UI")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

DEFAULT_SETUP_ROOT = "data/my_dataset"
DEFAULT_CLASSES = "person,car,bus"
DEFAULT_DATA_YAML = "data/my_dataset/data.yaml"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_DEVICE = ""
DEFAULT_PROJECT = "runs/detect"
DEFAULT_RUN_NAME = "train"
DEFAULT_CFG = ""
DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"
DEFAULT_EXPORT_BATCH = 1
DEFAULT_OPSET = 12


def list_paths(pattern: str, default_value: str) -> list[str]:
    options = {default_value}
    for path in REPO_ROOT.glob(pattern):
        if path.is_file():
            options.add(str(path.relative_to(REPO_ROOT)))
    return sorted(options)


def resolve_repo_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def render_page(request: Request, template_name: str, **context: Any):
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=context,
    )


@app.get("/")
async def index(request: Request):
    return render_page(request, "index.html", active_page="home")


@app.get("/setup")
async def setup_page(request: Request):
    return render_page(
        request,
        "setup.html",
        active_page="setup",
        form_data={
            "dataset_root": DEFAULT_SETUP_ROOT,
            "classes": DEFAULT_CLASSES,
        },
        message=None,
        message_type="info",
    )


@app.post("/setup")
async def run_setup(
    request: Request,
    dataset_root: Annotated[str, Form(...)],
    classes: Annotated[str, Form(...)],
):
    message = None
    message_type = "success"
    try:
        data_yaml_path = setup_dataset(root_path=dataset_root, classes_csv=classes)
        message = f"Created dataset structure and wrote: {data_yaml_path}"
    except Exception as exc:  # pragma: no cover - UI feedback path
        message = f"Operation failed: {exc}"
        message_type = "error"

    return render_page(
        request,
        "setup.html",
        active_page="setup",
        form_data={
            "dataset_root": dataset_root,
            "classes": classes,
        },
        message=message,
        message_type=message_type,
    )


@app.get("/train")
async def train_page(request: Request):
    return render_page(
        request,
        "train.html",
        active_page="train",
        form_data={
            "data_yaml": DEFAULT_DATA_YAML,
            "model_name": DEFAULT_MODEL,
            "epochs": DEFAULT_EPOCHS,
            "imgsz": DEFAULT_IMGSZ,
            "batch": DEFAULT_BATCH,
            "device": DEFAULT_DEVICE,
            "project": DEFAULT_PROJECT,
            "run_name": DEFAULT_RUN_NAME,
            "cfg": DEFAULT_CFG,
            "resume": False,
        },
        data_yaml_options=list_paths("**/data.yaml", DEFAULT_DATA_YAML),
        message=None,
        message_type="info",
    )


@app.post("/train")
async def run_train(
    request: Request,
    data_yaml: Annotated[str, Form(...)],
    model_name: Annotated[str, Form(...)],
    epochs: Annotated[int, Form(...)],
    imgsz: Annotated[int, Form(...)],
    batch: Annotated[int, Form(...)],
    project: Annotated[str, Form(...)],
    run_name: Annotated[str, Form(...)],
    device: Annotated[str, Form()] = "",
    cfg: Annotated[str, Form()] = "",
    resume: Annotated[bool, Form()] = False,
):
    message_type = "success"
    data_path = resolve_repo_path(data_yaml)

    if not data_path.exists():
        message = f"data.yaml not found: {data_path}"
        message_type = "error"
    else:
        try:
            train_model(
                data=str(data_path),
                model_name=model_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=run_name,
                resume=resume,
                device=device.strip() or None,
                cfg=cfg.strip() or None,
            )
            message = "Training completed successfully."
        except Exception as exc:  # pragma: no cover - UI feedback path
            message = f"Operation failed: {exc}"
            message_type = "error"

    return render_page(
        request,
        "train.html",
        active_page="train",
        form_data={
            "data_yaml": data_yaml,
            "model_name": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "project": project,
            "run_name": run_name,
            "cfg": cfg,
            "resume": resume,
        },
        data_yaml_options=list_paths("**/data.yaml", DEFAULT_DATA_YAML),
        message=message,
        message_type=message_type,
    )


@app.get("/export")
async def export_page(request: Request):
    return render_page(
        request,
        "export.html",
        active_page="export",
        form_data={
            "weights": DEFAULT_WEIGHTS,
            "imgsz": DEFAULT_IMGSZ,
            "batch": DEFAULT_EXPORT_BATCH,
            "opset": DEFAULT_OPSET,
            "dynamic": False,
        },
        weights_options=list_paths("**/*.pt", DEFAULT_WEIGHTS),
        message=None,
        message_type="info",
    )


@app.post("/export")
async def run_export(
    request: Request,
    weights: Annotated[str, Form(...)],
    imgsz: Annotated[int, Form(...)],
    batch: Annotated[int, Form(...)],
    opset: Annotated[int, Form(...)],
    dynamic: Annotated[bool, Form()] = False,
):
    message_type = "success"
    weights_path = resolve_repo_path(weights)

    if not weights_path.exists():
        message = f"Weights file not found: {weights_path}"
        message_type = "error"
    else:
        try:
            export_onnx(
                weights=str(weights_path),
                imgsz=imgsz,
                batch=batch,
                opset=opset,
                dynamic=dynamic,
            )
            message = "ONNX export completed."
        except Exception as exc:  # pragma: no cover - UI feedback path
            message = f"Operation failed: {exc}"
            message_type = "error"

    return render_page(
        request,
        "export.html",
        active_page="export",
        form_data={
            "weights": weights,
            "imgsz": imgsz,
            "batch": batch,
            "opset": opset,
            "dynamic": dynamic,
        },
        weights_options=list_paths("**/*.pt", DEFAULT_WEIGHTS),
        message=message,
        message_type=message_type,
    )
