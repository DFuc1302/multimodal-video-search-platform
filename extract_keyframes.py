"""Backward-compatible entry point for frame extraction."""

from pipeline.frames.extract import *  # noqa: F401,F403
from pipeline.frames.extract import main


if __name__ == "__main__":
    raise SystemExit(main())
