"""Compatibility wrapper to run the workshop CLI."""
from __future__ import annotations

from .main import main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    main()
