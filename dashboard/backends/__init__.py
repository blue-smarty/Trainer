from __future__ import annotations

from dashboard.backends.base import BackendAdapter
from dashboard.backends.custom_pytorch import CustomPytorchBackend
from dashboard.backends.image_classification import ImageClassificationBackend
from dashboard.backends.yolo_detection import YoloDetectionBackend


def get_backends() -> list[BackendAdapter]:
    return [
        YoloDetectionBackend(),
        ImageClassificationBackend(),
        CustomPytorchBackend(),
    ]


def get_backend_map() -> dict[str, BackendAdapter]:
    backends = get_backends()
    return {backend.key: backend for backend in backends}
