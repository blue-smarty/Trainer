from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from dashboard.backends.base import BackendAdapter
from dashboard.validation import ValidationResult


class CustomPyTorchBackend(BackendAdapter):
    key = "custom_pytorch"
    label = "Custom PyTorch Model"
    description = "Run user-defined PyTorch train/export functions from a Python module."

    def render_train_controls(self, repo_root: Path) -> dict[str, Any]:
        module_path = st.text_input(
            "Training module path",
            value="scripts/custom_model.py",
            help="Python file that defines the training function.",
            key="tr_custom_module",
        )
        function_name = st.text_input(
            "Training function name",
            value="train_model",
            help="Function called as train_model(config_dict).",
            key="tr_custom_fn",
        )
        config_text = st.text_area(
            "Training config (JSON)",
            value='{"epochs": 10, "device": "cpu"}',
            help="JSON object passed to your training function.",
            key="tr_custom_config",
        )
        return {
            "module_path": module_path,
            "function_name": function_name,
            "config_text": config_text,
        }

    def validate_train_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        return self._validate_common(config, repo_root, config_key="config_text")

    def train_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        module_path = self._resolve_module_path(config["module_path"], repo_root)
        payload = json.loads(config["config_text"])
        fn = self._load_callable(module_path, config["function_name"])
        output = self._invoke(fn, payload)
        return {"output": output}

    def render_export_controls(self, repo_root: Path) -> dict[str, Any]:
        module_path = st.text_input(
            "Export module path",
            value="scripts/custom_model.py",
            help="Python file that defines the export function.",
            key="ex_custom_module",
        )
        function_name = st.text_input(
            "Export function name",
            value="export_model",
            help="Function called as export_model(config_dict).",
            key="ex_custom_fn",
        )
        config_text = st.text_area(
            "Export config (JSON)",
            value='{"checkpoint": "runs/custom/model.pt", "output": "runs/custom/model.onnx"}',
            help="JSON object passed to your export function.",
            key="ex_custom_config",
        )
        return {
            "module_path": module_path,
            "function_name": function_name,
            "config_text": config_text,
        }

    def validate_export_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        return self._validate_common(config, repo_root, config_key="config_text")

    def export_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        module_path = self._resolve_module_path(config["module_path"], repo_root)
        payload = json.loads(config["config_text"])
        fn = self._load_callable(module_path, config["function_name"])
        output = self._invoke(fn, payload)
        return {"output": output}

    @staticmethod
    def _resolve_module_path(module_path: str, repo_root: Path) -> Path:
        path = Path(module_path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path

    def _validate_common(
        self, config: dict[str, Any], repo_root: Path, config_key: str
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if not config["module_path"]:
            errors.append("Module path must not be empty.")
        else:
            module_path = self._resolve_module_path(config["module_path"], repo_root)
            if not module_path.exists():
                errors.append(f"Module file not found: {module_path}")
        if not config["function_name"]:
            errors.append("Function name must not be empty.")
        try:
            payload = json.loads(config[config_key])
            if not isinstance(payload, dict):
                errors.append("Config JSON must be an object.")
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON config: {exc}")
            payload = None
        if payload == {}:
            warnings.append("Config JSON is empty.")
        return ValidationResult(errors=errors, warnings=warnings)

    @staticmethod
    def _load_callable(module_path: Path, function_name: str) -> Callable[..., Any]:
        spec = importlib.util.spec_from_file_location("custom_backend_module", str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, function_name, None)
        if not callable(fn):
            raise RuntimeError(f"Function '{function_name}' not found in {module_path}")
        return fn

    @staticmethod
    def _invoke(fn: Callable[..., Any], payload: dict[str, Any]) -> Any:
        params = list(inspect.signature(fn).parameters.values())
        if len(params) == 1 and params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return fn(payload)
        return fn(**payload)
