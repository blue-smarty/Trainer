#!/usr/bin/env python3
"""Train a YOLOv8 model for object detection.

Examples:
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8n.pt --epochs 50
  python scripts/train.py --data data/my_dataset/data.yaml --model yolov8s.pt --device cpu
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


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


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.cfg:
        train_kwargs["cfg"] = args.cfg

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
