"""Tests for scripts/setup_dataset.py"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.setup_dataset import setup_dataset


class TestSetupDataset:
    def test_creates_image_directories(self, tmp_path):
        setup_dataset(str(tmp_path / "mydata"), "cat,dog")
        root = tmp_path / "mydata"
        for split in ("train", "val", "test"):
            assert (root / "images" / split).is_dir()

    def test_creates_label_directories(self, tmp_path):
        setup_dataset(str(tmp_path / "mydata"), "cat,dog")
        root = tmp_path / "mydata"
        for split in ("train", "val", "test"):
            assert (root / "labels" / split).is_dir()

    def test_writes_data_yaml(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), "cat,dog")
        assert data_path.exists()
        assert data_path.name == "data.yaml"

    def test_data_yaml_contains_class_names(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), "cat,dog")
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert data["names"] == {0: "cat", 1: "dog"}

    def test_data_yaml_contains_split_paths(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), "cat")
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert data["test"] == "images/test"

    def test_data_yaml_contains_root_path(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), "cat")
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert "path" in data
        assert data["path"] == str(root.resolve())

    def test_classes_with_surrounding_whitespace_stripped(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), " cat , dog , bird ")
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert data["names"] == {0: "cat", 1: "dog", 2: "bird"}

    def test_single_class(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), "person")
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert data["names"] == {0: "person"}

    def test_many_classes(self, tmp_path):
        classes = ",".join(f"class{i}" for i in range(20))
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), classes)
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert len(data["names"]) == 20

    def test_empty_classes_raises_systemexit(self, tmp_path):
        with pytest.raises(SystemExit):
            setup_dataset(str(tmp_path / "mydata"), ",,,,")

    def test_empty_string_classes_raises_systemexit(self, tmp_path):
        with pytest.raises(SystemExit):
            setup_dataset(str(tmp_path / "mydata"), "")

    def test_returns_path_object(self, tmp_path):
        result = setup_dataset(str(tmp_path / "mydata"), "cat")
        assert isinstance(result, Path)

    def test_returns_path_with_yaml_suffix(self, tmp_path):
        result = setup_dataset(str(tmp_path / "mydata"), "cat")
        assert result.suffix == ".yaml"

    def test_idempotent_second_call_does_not_raise(self, tmp_path):
        root_str = str(tmp_path / "mydata")
        setup_dataset(root_str, "cat,dog")
        setup_dataset(root_str, "cat,dog,bird")

    def test_nested_root_path_created(self, tmp_path):
        deep_root = tmp_path / "a" / "b" / "c" / "dataset"
        setup_dataset(str(deep_root), "cat")
        assert (deep_root / "images" / "train").is_dir()

    def test_class_index_starts_at_zero(self, tmp_path):
        root = tmp_path / "mydata"
        data_path = setup_dataset(str(root), "alpha,beta,gamma")
        with data_path.open() as f:
            data = yaml.safe_load(f)
        assert 0 in data["names"]
        assert data["names"][0] == "alpha"
