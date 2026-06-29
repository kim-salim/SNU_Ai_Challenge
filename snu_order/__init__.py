"""Import shim for running `python -m snu_order...` from the repository root."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "snu_order"
if not _SRC_PACKAGE.exists():
    raise ImportError(f"Expected source package at {_SRC_PACKAGE}")

__path__ = [str(_SRC_PACKAGE)]
__version__ = "0.1.0"

