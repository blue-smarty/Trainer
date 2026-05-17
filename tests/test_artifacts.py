"""Tests for dashboard/artifacts.py"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dashboard.artifacts import (
    RunInfo,
    find_all_hef,
    find_all_onnx,
    find_recent_runs,
    format_mtime,
    format_size,
    infer_onnx_path,
)


# ---------------------------------------------------------------------------
# RunInfo
# ---------------------------------------------------------------------------


class TestRunInfo:
    def test_best_pt_found(self, tmp_path):
        weights = [tmp_path / "best.pt", tmp_path / "last.pt"]
        ri = RunInfo(path=tmp_path, name="run1", mtime=0.0, weights=weights)
        assert ri.best_pt == tmp_path / "best.pt"

    def test_best_pt_not_found(self, tmp_path):
        weights = [tmp_path / "last.pt"]
        ri = RunInfo(path=tmp_path, name="run1", mtime=0.0, weights=weights)
        assert ri.best_pt is None

    def test_last_pt_found(self, tmp_path):
        weights = [tmp_path / "best.pt", tmp_path / "last.pt"]
        ri = RunInfo(path=tmp_path, name="run1", mtime=0.0, weights=weights)
        assert ri.last_pt == tmp_path / "last.pt"

    def test_last_pt_not_found(self, tmp_path):
        weights = [tmp_path / "best.pt"]
        ri = RunInfo(path=tmp_path, name="run1", mtime=0.0, weights=weights)
        assert ri.last_pt is None

    def test_empty_weights_defaults(self, tmp_path):
        ri = RunInfo(path=tmp_path, name="run1", mtime=0.0)
        assert ri.best_pt is None
        assert ri.last_pt is None

    def test_default_lists_are_empty(self, tmp_path):
        ri = RunInfo(path=tmp_path, name="run1", mtime=0.0)
        assert ri.weights == []
        assert ri.onnx_files == []
        assert ri.hef_files == []


# ---------------------------------------------------------------------------
# find_recent_runs
# ---------------------------------------------------------------------------


class TestFindRecentRuns:
    def test_nonexistent_root_returns_empty(self, tmp_path):
        result = find_recent_runs(tmp_path / "nonexistent")
        assert result == []

    def test_empty_directory_returns_empty(self, tmp_path):
        result = find_recent_runs(tmp_path)
        assert result == []

    def test_files_in_root_are_ignored(self, tmp_path):
        (tmp_path / "somefile.txt").write_text("x")
        result = find_recent_runs(tmp_path)
        assert result == []

    def test_directory_without_weights_is_included(self, tmp_path):
        (tmp_path / "train1").mkdir()
        result = find_recent_runs(tmp_path)
        assert len(result) == 1
        assert result[0].weights == []

    def test_weights_are_collected(self, tmp_path):
        run_dir = tmp_path / "train1"
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").write_text("x")
        (run_dir / "weights" / "last.pt").write_text("x")
        result = find_recent_runs(tmp_path)
        assert len(result) == 1
        names = {p.name for p in result[0].weights}
        assert "best.pt" in names
        assert "last.pt" in names

    def test_non_pt_files_in_weights_excluded(self, tmp_path):
        run_dir = tmp_path / "run1"
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").write_text("x")
        (run_dir / "weights" / "notes.txt").write_text("x")
        result = find_recent_runs(tmp_path)
        assert all(p.suffix == ".pt" for p in result[0].weights)

    def test_sorted_newest_first(self, tmp_path):
        for i in range(3):
            d = tmp_path / f"run{i}"
            d.mkdir()
            os.utime(d, (i * 10.0, i * 10.0))
        result = find_recent_runs(tmp_path)
        mtimes = [r.mtime for r in result]
        assert mtimes == sorted(mtimes, reverse=True)

    def test_max_runs_limits_results(self, tmp_path):
        for i in range(5):
            (tmp_path / f"run{i}").mkdir()
        result = find_recent_runs(tmp_path, max_runs=3)
        assert len(result) == 3

    def test_default_max_runs_is_10(self, tmp_path):
        for i in range(12):
            (tmp_path / f"run{i:02d}").mkdir()
        result = find_recent_runs(tmp_path)
        assert len(result) == 10

    def test_onnx_files_collected(self, tmp_path):
        run_dir = tmp_path / "train1"
        run_dir.mkdir()
        (run_dir / "model.onnx").write_text("x")
        result = find_recent_runs(tmp_path)
        assert len(result[0].onnx_files) == 1
        assert result[0].onnx_files[0].name == "model.onnx"

    def test_hef_files_collected(self, tmp_path):
        run_dir = tmp_path / "train1"
        run_dir.mkdir()
        (run_dir / "model.hef").write_bytes(b"\x00")
        result = find_recent_runs(tmp_path)
        assert len(result[0].hef_files) == 1
        assert result[0].hef_files[0].name == "model.hef"

    def test_run_name_matches_directory_name(self, tmp_path):
        (tmp_path / "my_experiment").mkdir()
        result = find_recent_runs(tmp_path)
        assert result[0].name == "my_experiment"

    def test_accepts_string_path(self, tmp_path):
        (tmp_path / "run1").mkdir()
        result = find_recent_runs(str(tmp_path))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# find_all_onnx
# ---------------------------------------------------------------------------


class TestFindAllOnnx:
    def test_empty_directory(self, tmp_path):
        assert find_all_onnx(tmp_path) == []

    def test_finds_onnx_at_root(self, tmp_path):
        (tmp_path / "model.onnx").write_text("x")
        result = find_all_onnx(tmp_path)
        assert len(result) == 1
        assert result[0].suffix == ".onnx"

    def test_finds_onnx_in_subdirectory(self, tmp_path):
        sub = tmp_path / "runs" / "train"
        sub.mkdir(parents=True)
        (sub / "best.onnx").write_text("x")
        result = find_all_onnx(tmp_path)
        assert len(result) == 1

    def test_ignores_non_onnx_files(self, tmp_path):
        (tmp_path / "model.pt").write_text("x")
        (tmp_path / "model.hef").write_bytes(b"\x00")
        assert find_all_onnx(tmp_path) == []

    def test_accepts_string_path(self, tmp_path):
        (tmp_path / "a.onnx").write_text("x")
        result = find_all_onnx(str(tmp_path))
        assert len(result) == 1

    def test_multiple_files_returned(self, tmp_path):
        for name in ("a.onnx", "b.onnx", "c.onnx"):
            (tmp_path / name).write_text("x")
        result = find_all_onnx(tmp_path)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# find_all_hef
# ---------------------------------------------------------------------------


class TestFindAllHef:
    def test_empty_directory(self, tmp_path):
        assert find_all_hef(tmp_path) == []

    def test_finds_hef_files(self, tmp_path):
        (tmp_path / "a.hef").write_bytes(b"\x00")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.hef").write_bytes(b"\x00")
        result = find_all_hef(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".hef" for p in result)

    def test_ignores_non_hef_files(self, tmp_path):
        (tmp_path / "model.onnx").write_text("x")
        assert find_all_hef(tmp_path) == []


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_zero_bytes(self, tmp_path):
        f = tmp_path / "f"
        f.write_bytes(b"")
        assert format_size(f) == "0.0 B"

    def test_bytes_range(self, tmp_path):
        f = tmp_path / "f"
        f.write_bytes(b"x" * 512)
        assert format_size(f) == "512.0 B"

    def test_kib_range(self, tmp_path):
        f = tmp_path / "f"
        f.write_bytes(b"x" * 2048)
        assert format_size(f) == "2.0 KiB"

    def test_mib_range(self, tmp_path):
        mock_stat = MagicMock()
        mock_stat.st_size = 3 * 1024 * 1024
        with patch.object(Path, "stat", return_value=mock_stat):
            assert format_size(Path("/fake/file")) == "3.0 MiB"

    def test_gib_range(self, tmp_path):
        mock_stat = MagicMock()
        mock_stat.st_size = 2 * 1024 * 1024 * 1024
        with patch.object(Path, "stat", return_value=mock_stat):
            assert format_size(Path("/fake/file")) == "2.0 GiB"

    def test_oserror_returns_unknown(self, tmp_path):
        assert format_size(tmp_path / "nonexistent") == "unknown size"

    def test_boundary_exactly_1023_bytes(self, tmp_path):
        f = tmp_path / "f"
        f.write_bytes(b"x" * 1023)
        assert format_size(f) == "1023.0 B"

    def test_boundary_exactly_1024_bytes(self, tmp_path):
        f = tmp_path / "f"
        f.write_bytes(b"x" * 1024)
        assert format_size(f) == "1.0 KiB"


# ---------------------------------------------------------------------------
# format_mtime
# ---------------------------------------------------------------------------


class TestFormatMtime:
    def test_returns_formatted_datetime(self, tmp_path):
        f = tmp_path / "f"
        f.write_text("x")
        result = format_mtime(f)
        # Expect "YYYY-MM-DD HH:MM" (16 chars)
        assert len(result) == 16
        assert result[4] == "-"
        assert result[7] == "-"
        assert result[10] == " "
        assert result[13] == ":"

    def test_oserror_returns_unknown(self, tmp_path):
        assert format_mtime(tmp_path / "nonexistent") == "unknown time"


# ---------------------------------------------------------------------------
# infer_onnx_path
# ---------------------------------------------------------------------------


class TestInferOnnxPath:
    def test_returns_onnx_when_exists(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        onnx = tmp_path / "best.onnx"
        onnx.write_text("x")
        assert infer_onnx_path(pt) == onnx

    def test_returns_none_when_onnx_absent(self, tmp_path):
        pt = tmp_path / "best.pt"
        pt.write_text("x")
        assert infer_onnx_path(pt) is None

    def test_stem_preserved(self, tmp_path):
        pt = tmp_path / "epoch100.pt"
        onnx = tmp_path / "epoch100.onnx"
        onnx.write_text("x")
        assert infer_onnx_path(pt) == onnx
