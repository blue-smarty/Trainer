"""Tests for pure helper functions in scripts/onnx_to_hef."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.onnx_to_hef import (
    VALID_HW_ARCHS,
    _extract_expected_input_shape,
    _extract_suggested_end_nodes,
    _load_calibration_images,
    _parse_node_names,
    _permute_calib_to_expected_shape,
    convert_onnx_to_hef,
)


# ---------------------------------------------------------------------------
# VALID_HW_ARCHS
# ---------------------------------------------------------------------------


class TestValidHwArchs:
    def test_is_tuple(self):
        assert isinstance(VALID_HW_ARCHS, tuple)

    def test_contains_hailo8(self):
        assert "hailo8" in VALID_HW_ARCHS

    def test_contains_hailo8l(self):
        assert "hailo8l" in VALID_HW_ARCHS

    def test_contains_hailo8r(self):
        assert "hailo8r" in VALID_HW_ARCHS

    def test_not_empty(self):
        assert len(VALID_HW_ARCHS) > 0


# ---------------------------------------------------------------------------
# _parse_node_names
# ---------------------------------------------------------------------------


class TestParseNodeNames:
    def test_none_returns_empty(self):
        assert _parse_node_names(None) == []

    def test_empty_list_returns_empty(self):
        assert _parse_node_names([]) == []

    def test_strips_whitespace(self):
        assert _parse_node_names(["  /node1  ", " /node2"]) == ["/node1", "/node2"]

    def test_drops_empty_strings(self):
        assert _parse_node_names(["", "  ", "/real"]) == ["/real"]

    def test_drops_whitespace_only_entries(self):
        assert _parse_node_names(["   ", "   "]) == []

    def test_single_entry(self):
        assert _parse_node_names(["/output"]) == ["/output"]

    def test_preserves_order(self):
        nodes = ["/a", "/b", "/c"]
        assert _parse_node_names(nodes) == nodes

    def test_mixed_valid_and_empty(self):
        result = _parse_node_names(["/a", "", "/b", "  "])
        assert result == ["/a", "/b"]


# ---------------------------------------------------------------------------
# _extract_suggested_end_nodes
# ---------------------------------------------------------------------------


class TestExtractSuggestedEndNodes:
    def test_empty_string_returns_empty(self):
        assert _extract_suggested_end_nodes("") == []

    def test_no_match_returns_empty(self):
        assert _extract_suggested_end_nodes("some random error text") == []

    def test_single_node_extracted(self):
        err = "Parse failed; using these end node names: /model/output"
        result = _extract_suggested_end_nodes(err)
        assert result == ["/model/output"]

    def test_multiple_nodes_extracted(self):
        err = "Error using these end node names: /a, /b, /c\nmore info"
        result = _extract_suggested_end_nodes(err)
        assert result == ["/a", "/b", "/c"]

    def test_case_insensitive_match(self):
        err = "Using These End Node Names: /x"
        result = _extract_suggested_end_nodes(err)
        assert result == ["/x"]

    def test_whitespace_trimmed_from_nodes(self):
        err = "using these end node names:  /node1 ,  /node2 "
        result = _extract_suggested_end_nodes(err)
        assert result == ["/node1", "/node2"]

    def test_nodes_with_slashes_and_numbers(self):
        err = "using these end node names: /model/22, /model/23"
        result = _extract_suggested_end_nodes(err)
        assert result == ["/model/22", "/model/23"]


# ---------------------------------------------------------------------------
# _extract_expected_input_shape
# ---------------------------------------------------------------------------


class TestExtractExpectedInputShape:
    def test_no_match_returns_none(self):
        assert _extract_expected_input_shape("no shape here") is None

    def test_empty_string_returns_none(self):
        assert _extract_expected_input_shape("") is None

    def test_extracts_nchw_shape(self):
        err = "shape mismatch: network's input shape (1, 3, 640, 640)"
        assert _extract_expected_input_shape(err) == (1, 3, 640, 640)

    def test_extracts_nhwc_shape(self):
        err = "network's input shape (1, 640, 640, 3)"
        assert _extract_expected_input_shape(err) == (1, 640, 640, 3)

    def test_extracts_3d_shape(self):
        err = "network's input shape (3, 224, 224)"
        assert _extract_expected_input_shape(err) == (3, 224, 224)

    def test_returns_tuple_of_ints(self):
        err = "network's input shape (1, 3, 224, 224)"
        result = _extract_expected_input_shape(err)
        assert isinstance(result, tuple)
        assert all(isinstance(v, int) for v in result)


# ---------------------------------------------------------------------------
# _permute_calib_to_expected_shape
# ---------------------------------------------------------------------------


def _make_calib(sample_shape: tuple[int, ...], n: int = 4) -> np.ndarray:
    return np.zeros((n,) + sample_shape, dtype=np.float32)


class TestPermuteCalibToExpectedShape:
    def test_identity_returns_same_object(self):
        calib = _make_calib((3, 4, 5))
        result = _permute_calib_to_expected_shape(calib, (3, 4, 5))
        assert result is calib

    def test_basic_transpose(self):
        calib = _make_calib((3, 4, 5))
        result = _permute_calib_to_expected_shape(calib, (5, 3, 4))
        assert result is not None
        assert result.shape == (4, 5, 3, 4)

    def test_hwc_to_chw(self):
        # Typical NHWC → NCHW for Hailo
        calib = _make_calib((640, 640, 3))
        result = _permute_calib_to_expected_shape(calib, (3, 640, 640))
        assert result is not None
        assert result.shape[1:] == (3, 640, 640)

    def test_chw_to_hwc(self):
        calib = _make_calib((3, 64, 64))
        result = _permute_calib_to_expected_shape(calib, (64, 64, 3))
        assert result is not None
        assert result.shape[1:] == (64, 64, 3)

    def test_incompatible_rank_returns_none(self):
        calib = _make_calib((3, 4))
        result = _permute_calib_to_expected_shape(calib, (3, 4, 5))
        assert result is None

    def test_incompatible_dimensions_returns_none(self):
        calib = _make_calib((3, 4, 5))
        # 6 does not appear in sample_shape → no permutation possible
        result = _permute_calib_to_expected_shape(calib, (3, 4, 6))
        assert result is None

    def test_too_many_sample_dims_returns_none(self):
        # Rank-5 sample (beyond the 4-dim guard)
        calib = _make_calib((2, 3, 4, 5, 6))
        result = _permute_calib_to_expected_shape(calib, (2, 3, 4, 5, 6))
        assert result is None

    def test_permuted_data_values_preserved(self):
        # Create a simple gradient array and check values after permutation
        calib = np.arange(4 * 3 * 4 * 5, dtype=np.float32).reshape(4, 3, 4, 5)
        result = _permute_calib_to_expected_shape(calib, (5, 3, 4))
        assert result is not None
        # Verify via numpy that the permutation is consistent
        expected = calib.transpose(0, 3, 1, 2)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _load_calibration_images
# ---------------------------------------------------------------------------


class TestLoadCalibrationImages:
    def test_raises_not_a_directory_error(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            _load_calibration_images(str(f), 64, 64)

    def test_raises_file_not_found_for_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No images found"):
            _load_calibration_images(str(tmp_path), 64, 64)

    def test_raises_file_not_found_for_no_image_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        with pytest.raises(FileNotFoundError, match="No images found"):
            _load_calibration_images(str(tmp_path), 64, 64)

    def test_raises_value_error_for_all_corrupt_images(self, tmp_path):
        (tmp_path / "bad.png").write_bytes(b"notanimage")
        with pytest.raises(ValueError, match="Could not decode"):
            _load_calibration_images(str(tmp_path), 64, 64)

    def test_loads_png_image(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "test.png"), img)
        result = _load_calibration_images(str(tmp_path), 16, 16)
        assert result.shape == (1, 16, 16, 3)
        assert result.dtype == np.float32

    def test_normalises_to_0_1(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        # Pure white image (all 255)
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "white.png"), img)
        result = _load_calibration_images(str(tmp_path), 32, 32)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_resizes_to_requested_dimensions(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "big.png"), img)
        result = _load_calibration_images(str(tmp_path), 32, 48)
        assert result.shape[1] == 32  # height
        assert result.shape[2] == 48  # width

    def test_multiple_images_stacked(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        for i in range(3):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img{i}.png"), img)
        result = _load_calibration_images(str(tmp_path), 16, 16)
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# convert_onnx_to_hef – import / file-existence guards
# ---------------------------------------------------------------------------


class TestConvertOnnxToHef:
    def test_import_error_when_hailo_sdk_not_installed(self, tmp_path, monkeypatch):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        # Force hailo_sdk_client to be absent
        monkeypatch.setitem(sys.modules, "hailo_sdk_client", None)
        with pytest.raises(ImportError, match="hailo_sdk_client"):
            convert_onnx_to_hef(str(onnx), hw_arch="hailo8l")

    def test_file_not_found_when_hailo_sdk_present(self, tmp_path, monkeypatch):
        # Mock hailo_sdk_client so the import succeeds
        mock_hailo = MagicMock()
        monkeypatch.setitem(sys.modules, "hailo_sdk_client", mock_hailo)
        missing = tmp_path / "missing.onnx"
        with pytest.raises(FileNotFoundError, match="missing.onnx"):
            convert_onnx_to_hef(str(missing), hw_arch="hailo8l")

    def test_import_error_message_is_helpful(self, tmp_path, monkeypatch):
        onnx = tmp_path / "model.onnx"
        onnx.write_text("x")
        monkeypatch.setitem(sys.modules, "hailo_sdk_client", None)
        with pytest.raises(ImportError) as exc_info:
            convert_onnx_to_hef(str(onnx), hw_arch="hailo8l")
        assert "developer.hailo.ai" in str(exc_info.value).lower() or \
               "hailo" in str(exc_info.value).lower()
