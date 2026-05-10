# Trainer

A repository for training object detection models (YOLOv8/YOLO11/PyTorch) for Raspberry Pi 5 and Hailo-8/8L.

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

You can choose any Ultralytics-compatible model with `--model`. Common options:

| Model | Size | Notes |
|-------|------|-------|
| `yolov8n.pt` | Nano | Fastest, least accurate |
| `yolov8s.pt` | Small | Good balance |
| `yolov8m.pt` | Medium | Better accuracy |
| `yolov8l.pt` | Large | High accuracy |
| `yolov8x.pt` | XLarge | Most accurate |
| `yolo11n.pt` | Nano (v11) | Latest nano |
| `yolo11s.pt` | Small (v11) | Latest small |

Use `--device` to select the training hardware (PyTorch/Ultralytics backends only):

```bash
# Train on CPU
python scripts/train.py --data data/my_dataset/data.yaml --model yolov8s.pt --device cpu

# Train on first CUDA GPU
python scripts/train.py --data data/my_dataset/data.yaml --model yolov8s.pt --device 0

# Train on Apple Silicon (MPS)
python scripts/train.py --data data/my_dataset/data.yaml --model yolov8s.pt --device mps
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

The dashboard provides a simple interface for:
- creating the dataset structure (`setup_dataset.py`)
- running training (`train.py`) — includes model and PyTorch device selectors
- exporting to ONNX (`export_hailo.py`)

## Hailo-8/8L Notes

- Use the **latest Hailo SDK/Dataflow Compiler** that supports Hailo-8/8L.
- After exporting to ONNX, compile to a HEF using Hailo's tools and calibration dataset.
- Follow Hailo's official documentation for compiler and runtime usage.
- Training runs on PyTorch-compatible devices (`cpu`, CUDA indices, or `mps`).
- For Hailo-8/8L deployment, train first and then export your trained `.pt` weights to ONNX with `scripts/export_hailo.py`.

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
