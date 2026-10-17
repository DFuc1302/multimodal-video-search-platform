"""Backward-compatible entry point for storage catalog generation."""

from pipeline.index.catalog import *  # noqa: F401,F403
from pipeline.index.catalog import main


if __name__ == "__main__":
    raise SystemExit(main())
