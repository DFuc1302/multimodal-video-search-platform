"""Backward-compatible entry point for serving bundle preparation."""

from pipeline.index.bundle import *  # noqa: F401,F403
from pipeline.index.bundle import main


if __name__ == "__main__":
    raise SystemExit(main())
