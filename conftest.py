"""Root conftest.py – adds the repository root to sys.path so that
``dashboard`` and ``scripts`` are importable as namespace packages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
