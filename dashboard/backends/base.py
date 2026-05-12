from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.validation import ValidationResult


def list_repo_paths(repo_root: Path, pattern: str, default_value: str) -> list[str]:
    options: set[str] = {default_value}
    for path in repo_root.glob(pattern):
        if path.is_file():
            options.add(str(path.relative_to(repo_root)))
    return sorted(options)


class BackendAdapter(ABC):
    key: str
    label: str
    description: str
    setup_supported: bool = False

    def get_runs_root(self, repo_root: Path) -> Path | None:
        return None

    def render_setup(self, repo_root: Path) -> None:
        st.info("Dataset setup is not implemented for this backend yet.")

    @abstractmethod
    def render_train_controls(self, repo_root: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_train_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        raise NotImplementedError

    @abstractmethod
    def train_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def render_export_controls(self, repo_root: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_export_config(self, config: dict[str, Any], repo_root: Path) -> ValidationResult:
        raise NotImplementedError

    @abstractmethod
    def export_model(self, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
        raise NotImplementedError
