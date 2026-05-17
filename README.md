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

This produces an ONNX file ready for the next step.

### 5) Convert ONNX to HEF (Hailo Dataflow Compiler)

> **Prerequisites:** Install the Hailo SDK (`hailo_sdk_client`) from the
> [Hailo Developer Zone](https://developer.hailo.ai).

```bash
# Hailo-8L (Raspberry Pi 5 AI HAT+)
python scripts/onnx_to_hef.py --onnx runs/detect/train/weights/best.onnx --hw-arch hailo8l

# With calibration images for better INT8 quantization accuracy
python scripts/onnx_to_hef.py --onnx best.onnx --hw-arch hailo8l --calib-path data/calib_images
```

This produces a `.hef` file in the same directory as the ONNX file. Use
`--output-dir` to write it elsewhere.

### 6) Launch the GUI dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides a simple interface for:
- creating the dataset structure (`setup_dataset.py`)
- running training (`train.py`)
- exporting to ONNX (`export_hailo.py`)
- converting ONNX to HEF (`onnx_to_hef.py`)

## Hailo-8/8L Notes

- Use the **latest Hailo SDK/Dataflow Compiler** that supports Hailo-8/8L.
- Export to ONNX with `export_hailo.py`, then compile to HEF with `onnx_to_hef.py`.
- Providing representative calibration images with `--calib-path` gives the best INT8 quantization accuracy.
- Follow Hailo's official documentation for runtime deployment.

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
  onnx_to_hef.py
```
