# Trainer

A repository for training object detection models (YOLOv8/PyTorch) for Raspberry Pi 5 and Hailo-8/8L.

## Quickstart

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note (Raspberry Pi 5):** Install a compatible PyTorch wheel for ARM64 before `pip install -r requirements.txt` if needed. The Ultralytics package depends on PyTorch.

### 2) Create dataset structure

```bash
python scripts/setup_dataset.py --root data/my_dataset --classes person,car,bus
```

This creates the standard YOLOv8 structure and a `data.yaml` for training.

### 3) Train (on host or on the Pi)

```bash
python scripts/train.py --data data/my_dataset/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

### 4) Export to ONNX (for Hailo toolchain)

```bash
python scripts/export_hailo.py --weights runs/detect/train/weights/best.pt --imgsz 640
```

This produces an ONNX file that you can compile with the Hailo Dataflow Compiler for Hailo-8/8L.

### 5) Launch the GUI dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides a backend/task selector with workflows for:
- YOLO Detection (dataset setup + training + ONNX export)
- Image Classification (training + ONNX export)
- Custom PyTorch Model (module-based training + export hooks)

## Hailo-8/8L Notes

- Use the **latest Hailo SDK/Dataflow Compiler** that supports Hailo-8/8L.
- After exporting to ONNX, compile to a HEF using Hailo’s tools and calibration dataset.
- Follow Hailo’s official documentation for compiler and runtime usage.

## Repository Layout

```
configs/
  yolov8-default.yaml
dashboard/
  app.py
scripts/
  setup_dataset.py
  train.py
  export_hailo.py
```
