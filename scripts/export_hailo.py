#!/usr/bin/env python3
"""Export a YOLOv8 model to ONNX for Hailo compilation.

Example:
  python scripts/export_hailo.py --weights runs/detect/train/weights/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--weights", required=True, help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=True,
    )


if __name__ == "__main__":
    main()
