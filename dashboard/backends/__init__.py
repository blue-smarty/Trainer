from __future__ import annotations

from dashboard.backends.base import BackendAdapter
from dashboard.backends.custom_pytorch import CustomPyTorchBackend
from dashboard.backends.image_classification import ImageClassificationBackend
from dashboard.backends.yolo_detection import YOLODetectionBackend


def get_backends() -> list[BackendAdapter]:
    return [
        YOLODetectionBackend(),
        ImageClassificationBackend(),
        CustomPyTorchBackend(),
    ]


def get_backend_map() -> dict[str, BackendAdapter]:
    backends = get_backends()
    return {backend.key: backend for backend in backends}
