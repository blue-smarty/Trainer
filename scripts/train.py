#!/usr/bin/env python3
"""Train a YOLO model for object detection.

Examples:
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8n.pt --epochs 50
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8s.pt --device cpu
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8m.pt --device mps
"""

from __future__ import annotations

import argparse

# Common YOLO models available via Ultralytics
SUPPORTED_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov8n-cls.pt",
    "yolov8s-cls.pt",
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]

# Valid string device options (in addition to CUDA indices like 0, 1 or 0,1)
SUPPORTED_DEVICES = ["cpu", "mps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Model name or path (e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt, "
            "yolov8l.pt, yolov8x.pt, yolo11n.pt). "
            "Can be any Ultralytics-compatible model file or pretrained name."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Training device: CUDA index (e.g. 0, 1, 0,1), 'cpu', or 'mps'. "
            "Defaults to auto-detect."
        ),
    )
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cfg", default=None, help="Optional Ultralytics config yaml")
    return parser.parse_args()


def train_model(
    data: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
    resume: bool = False,
    device: str | None = None,
    cfg: str | None = None,
) -> None:
    if device is not None and device.strip().lower() == "hailo8":
        raise ValueError(
            "Invalid training device 'hailo8'. Use a PyTorch/Ultralytics device "
            "such as 'cpu', 'mps', or CUDA indices (for example '0' or '0,1'). "
            "For Hailo-8/8L, train first and then export to ONNX via scripts/export_hailo.py."
        )

    from ultralytics import YOLO

    model = YOLO(model_name)
    train_kwargs = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": name,
        "resume": resume,
    }

    if device is not None:
        train_kwargs["device"] = device

    if cfg:
        train_kwargs["cfg"] = cfg

    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    train_model(
        data=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        resume=args.resume,
        device=args.device,
        cfg=args.cfg,
    )


if __name__ == "__main__":
    main()
