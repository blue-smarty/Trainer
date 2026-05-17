"""Tests for dashboard/validation.py"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dashboard.validation import (
    ValidationResult,
    validate_export_params,
    validate_hef_params,
    validate_setup_params,
    validate_train_params,
)
from scripts.onnx_to_hef import VALID_HW_ARCHS


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_ok_when_no_errors(self):
        vr = ValidationResult(errors=[], warnings=[])
        assert vr.ok is True

    def test_not_ok_with_errors(self):
        vr = ValidationResult(errors=["something went wrong"], warnings=[])
        assert vr.ok is False

    def test_ok_with_warnings_only(self):
        vr = ValidationResult(errors=[], warnings=["heads up"])
        assert vr.ok is True

    def test_multiple_errors(self):
        vr = ValidationResult(errors=["e1", "e2"], warnings=[])
        assert vr.ok is False


# ---------------------------------------------------------------------------
# validate_setup_params
# ---------------------------------------------------------------------------


class TestValidateSetupParams:
    def test_valid_inputs(self):
        r = validate_setup_params("/some/path", "cat,dog")
        assert r.ok

    def test_empty_root_path_is_error(self):
        r = validate_setup_params("", "cat,dog")
        assert not r.ok
        assert any("path" in e.lower() for e in r.errors)

    def test_whitespace_only_root_path_is_error(self):
        r = validate_setup_params("   ", "cat,dog")
        assert not r.ok

    def test_empty_classes_is_error(self):
        r = validate_setup_params("/some/path", "")
        assert not r.ok
        assert any("class" in e.lower() for e in r.errors)

    def test_whitespace_only_classes_is_error(self):
        r = validate_setup_params("/some/path", "  ,  ,  ")
        assert not r.ok

    def test_duplicate_classes_produce_warning(self):
        r = validate_setup_params("/some/path", "cat,dog,cat")
        assert r.ok
        assert any("duplicate" in w.lower() for w in r.warnings)

    def test_special_characters_produce_warning(self):
        r = validate_setup_params("/some/path", "cat$,dog")
        assert r.ok
        assert any("special" in w.lower() for w in r.warnings)

    def test_underscores_and_hyphens_are_valid(self):
        r = validate_setup_params("/some/path", "my_class,another-class")
        assert r.ok
        assert r.warnings == []

    def test_single_class_valid(self):
        r = validate_setup_params("/some/path", "person")
        assert r.ok

    def test_both_paths_empty_accumulates_both_errors(self):
        r = validate_setup_params("", "")
        assert not r.ok
        assert len(r.errors) >= 2


# ---------------------------------------------------------------------------
# validate_train_params – helpers
# ---------------------------------------------------------------------------


def _write_valid_yaml(tmp_path: Path, names: dict | None = None) -> str:
    if names is None:
        names = {0: "cat"}
    data_yaml = tmp_path / "data.yaml"
    content = {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(content))
    return str(data_yaml)


# ---------------------------------------------------------------------------
# validate_train_params
# ---------------------------------------------------------------------------


class TestValidateTrainParams:
    def test_valid_params(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, 16, "runs/detect")
        assert r.ok

    def test_empty_data_yaml_is_error(self):
        r = validate_train_params("", "yolov8n.pt", 50, 640, 16, "runs/detect")
        assert not r.ok
        assert any("data.yaml" in e.lower() or "path" in e.lower() for e in r.errors)

    def test_whitespace_data_yaml_is_error(self):
        r = validate_train_params("   ", "yolov8n.pt", 50, 640, 16, "runs/detect")
        assert not r.ok

    def test_missing_yaml_file_is_error(self, tmp_path):
        r = validate_train_params(
            str(tmp_path / "nosuchfile.yaml"), "yolov8n.pt", 50, 640, 16, "runs"
        )
        assert not r.ok
        assert any("not found" in e.lower() for e in r.errors)

    def test_empty_model_name_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "", 50, 640, 16, "runs/detect")
        assert not r.ok
        assert any("model" in e.lower() for e in r.errors)

    def test_epochs_zero_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 0, 640, 16, "runs/detect")
        assert not r.ok
        assert any("epoch" in e.lower() for e in r.errors)

    def test_epochs_negative_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", -5, 640, 16, "runs/detect")
        assert not r.ok

    def test_epochs_at_boundary_1_is_valid(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 1, 640, 16, "runs/detect")
        assert r.ok

    def test_very_high_epochs_produces_warning(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 10001, 640, 16, "runs/detect")
        assert r.ok
        assert any("epoch" in w.lower() for w in r.warnings)

    def test_imgsz_below_32_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 16, 16, "runs/detect")
        assert not r.ok
        assert any("image size" in e.lower() or "32" in e for e in r.errors)

    def test_imgsz_not_multiple_of_32_produces_warning(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 641, 16, "runs/detect")
        assert r.ok
        assert any("32" in w for w in r.warnings)

    def test_imgsz_exact_multiple_of_32_no_warning(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, 16, "runs/detect")
        assert r.ok
        assert not any("32" in w for w in r.warnings)

    def test_batch_zero_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, 0, "runs/detect")
        assert not r.ok
        assert any("batch" in e.lower() for e in r.errors)

    def test_batch_negative_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, -1, "runs/detect")
        assert not r.ok

    def test_empty_project_is_error(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, 16, "")
        assert not r.ok
        assert any("project" in e.lower() for e in r.errors)

    def test_data_yaml_missing_required_keys(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({"path": "/x", "train": "images/train"}))
        r = validate_train_params(str(bad), "yolov8n.pt", 50, 640, 16, "runs")
        assert not r.ok
        assert any("missing" in e.lower() for e in r.errors)

    def test_data_yaml_empty_names_is_error(self, tmp_path):
        bad = tmp_path / "data.yaml"
        bad.write_text(
            yaml.safe_dump({"path": "/x", "train": "t", "val": "v", "names": {}})
        )
        r = validate_train_params(str(bad), "yolov8n.pt", 50, 640, 16, "runs")
        assert not r.ok
        assert any("empty" in e.lower() or "names" in e.lower() for e in r.errors)

    def test_data_yaml_names_not_dict_is_error(self, tmp_path):
        bad = tmp_path / "data.yaml"
        bad.write_text(
            yaml.safe_dump(
                {"path": "/x", "train": "t", "val": "v", "names": ["cat", "dog"]}
            )
        )
        r = validate_train_params(str(bad), "yolov8n.pt", 50, 640, 16, "runs")
        assert not r.ok
        assert any("dict" in e.lower() or "names" in e.lower() for e in r.errors)

    def test_data_yaml_not_a_mapping_is_error(self, tmp_path):
        bad = tmp_path / "data.yaml"
        bad.write_text("- item1\n- item2\n")
        r = validate_train_params(str(bad), "yolov8n.pt", 50, 640, 16, "runs")
        assert not r.ok

    def test_data_yaml_unparseable_is_error(self, tmp_path):
        bad = tmp_path / "data.yaml"
        bad.write_bytes(b"\xff\xfe invalid yaml \x00")
        r = validate_train_params(str(bad), "yolov8n.pt", 50, 640, 16, "runs")
        assert not r.ok
        assert any("parse" in e.lower() or "yaml" in e.lower() for e in r.errors)

    def test_missing_split_paths_produce_warnings(self, tmp_path):
        # images/train and images/val don't actually exist → warnings
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, 16, "runs")
        assert r.ok
        assert len(r.warnings) >= 1

    def test_existing_split_paths_no_warnings(self, tmp_path):
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "images" / "val").mkdir(parents=True)
        yp = _write_valid_yaml(tmp_path)
        r = validate_train_params(yp, "yolov8n.pt", 50, 640, 16, "runs")
        assert r.ok
        assert r.warnings == []

    def test_relative_yaml_resolved_with_repo_root(self, tmp_path):
        yp = _write_valid_yaml(tmp_path)
        rel = Path(yp).relative_to(tmp_path)
        r = validate_train_params(
            str(rel), "yolov8n.pt", 50, 640, 16, "runs", repo_root=tmp_path
        )
        assert r.ok


# ---------------------------------------------------------------------------
# validate_hef_params
# ---------------------------------------------------------------------------


class TestValidateHefParams:
    def test_valid_params(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        calib = tmp_path / "calib"
        calib.mkdir()
        r = validate_hef_params(str(onnx), "hailo8l", str(calib))
        assert r.ok

    def test_empty_onnx_path_is_error(self):
        r = validate_hef_params("", "hailo8l", "")
        assert not r.ok
        assert any("onnx" in e.lower() for e in r.errors)

    def test_whitespace_onnx_path_is_error(self):
        r = validate_hef_params("   ", "hailo8l", "")
        assert not r.ok

    def test_missing_onnx_file_is_error(self, tmp_path):
        r = validate_hef_params(str(tmp_path / "missing.onnx"), "hailo8l", "")
        assert not r.ok
        assert any("not found" in e.lower() for e in r.errors)

    def test_wrong_extension_produces_warning(self, tmp_path):
        f = tmp_path / "model.pt"
        f.write_text("x")
        r = validate_hef_params(str(f), "hailo8l", "")
        assert r.ok
        assert any(".onnx" in w.lower() or "extension" in w.lower() for w in r.warnings)

    def test_invalid_hw_arch_is_error(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        r = validate_hef_params(str(onnx), "hailo999", "")
        assert not r.ok
        assert any("architecture" in e.lower() or "hailo999" in e for e in r.errors)

    @pytest.mark.parametrize("arch", list(VALID_HW_ARCHS))
    def test_all_valid_archs_pass(self, tmp_path, arch):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        r = validate_hef_params(str(onnx), arch, "")
        assert r.ok, f"Expected arch '{arch}' to be valid"

    def test_calib_not_a_directory_is_error(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        calib_file = tmp_path / "calib.txt"
        calib_file.write_text("x")
        r = validate_hef_params(str(onnx), "hailo8l", str(calib_file))
        assert not r.ok
        assert any("not a directory" in e.lower() for e in r.errors)

    def test_calib_missing_directory_is_error(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        r = validate_hef_params(str(onnx), "hailo8l", str(tmp_path / "nosuchdir"))
        assert not r.ok
        assert any("not found" in e.lower() for e in r.errors)

    def test_empty_calib_path_is_ignored(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        r = validate_hef_params(str(onnx), "hailo8l", "")
        assert r.ok

    def test_whitespace_calib_path_is_ignored(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        r = validate_hef_params(str(onnx), "hailo8l", "   ")
        assert r.ok

    def test_relative_onnx_resolved_with_repo_root(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        rel = onnx.relative_to(tmp_path)
        r = validate_hef_params(str(rel), "hailo8l", "", repo_root=tmp_path)
        assert r.ok

    def test_relative_calib_resolved_with_repo_root(self, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        calib = tmp_path / "calib"
        calib.mkdir()
        rel_calib = calib.relative_to(tmp_path)
        r = validate_hef_params(str(onnx), "hailo8l", str(rel_calib), repo_root=tmp_path)
        assert r.ok


# ---------------------------------------------------------------------------
# validate_export_params
# ---------------------------------------------------------------------------


class TestValidateExportParams:
    def test_valid_params(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 1, 12)
        assert r.ok

    def test_empty_weights_is_error(self):
        r = validate_export_params("", 640, 1, 12)
        assert not r.ok
        assert any("weights" in e.lower() for e in r.errors)

    def test_whitespace_weights_is_error(self):
        r = validate_export_params("   ", 640, 1, 12)
        assert not r.ok

    def test_missing_weights_file_is_error(self, tmp_path):
        r = validate_export_params(str(tmp_path / "no.pt"), 640, 1, 12)
        assert not r.ok
        assert any("not found" in e.lower() for e in r.errors)

    def test_wrong_extension_produces_warning(self, tmp_path):
        f = tmp_path / "model.onnx"
        f.write_text("x")
        r = validate_export_params(str(f), 640, 1, 12)
        assert r.ok
        assert any(".pt" in w.lower() or "extension" in w.lower() for w in r.warnings)

    def test_imgsz_below_32_is_error(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 16, 1, 12)
        assert not r.ok

    def test_imgsz_not_multiple_of_32_produces_warning(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 641, 1, 12)
        assert r.ok
        assert any("32" in w for w in r.warnings)

    def test_imgsz_exact_multiple_of_32_no_warning(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 1, 12)
        assert not any("32" in w for w in r.warnings)

    def test_batch_zero_is_error(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 0, 12)
        assert not r.ok
        assert any("batch" in e.lower() for e in r.errors)

    def test_batch_negative_is_error(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, -1, 12)
        assert not r.ok

    def test_opset_below_9_is_error(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 1, 8)
        assert not r.ok
        assert any("opset" in e.lower() for e in r.errors)

    def test_opset_at_boundary_9_is_valid(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 1, 9)
        assert r.ok

    def test_opset_above_18_produces_warning(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 1, 19)
        assert r.ok
        assert any("opset" in w.lower() for w in r.warnings)

    def test_opset_at_boundary_18_no_warning(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        r = validate_export_params(str(pt), 640, 1, 18)
        assert r.ok
        assert not any("opset" in w.lower() for w in r.warnings)

    def test_relative_weights_resolved_with_repo_root(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        rel = pt.relative_to(tmp_path)
        r = validate_export_params(str(rel), 640, 1, 12, repo_root=tmp_path)
        assert r.ok
