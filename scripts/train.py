#!/usr/bin/env python3
"""Train a YOLOv8 model for object detection.

Examples:
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8n.pt --epochs 50
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8s.pt --device cpu
"""

from __future__ import annotations

import argparse


def find_gpu() -> str | None:
    """Return a recommended training device string if a GPU is available."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    return "0"


def should_fallback_to_cpu(exc: Exception) -> bool:
    """Return True when the training error suggests retrying on CPU."""
    message = str(exc).lower()
    fallback_markers = (
        "cudaerrormemoryallocation",
        "out of memory",
        "cuda out of memory",
        "not enough memory",
        "all cuda-capable devices are busy",
        "device busy",
        "cuda error",
        "no cuda gpus are available",
        "cuda driver",
        "initialization error",
    )
    return any(marker in message for marker in fallback_markers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda device index, 'cpu', or None")
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

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        if device == "cpu" or not should_fallback_to_cpu(exc):
            raise

        print(
            "GPU training failed; retrying on CPU. "
            f"Original error: {exc}"
        )
        fallback_kwargs = dict(train_kwargs)
        fallback_kwargs["device"] = "cpu"
        model.train(**fallback_kwargs)


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
        device=args.device if args.device is not None else find_gpu(),
        cfg=args.cfg,
    )


if __name__ == "__main__":
    main()
