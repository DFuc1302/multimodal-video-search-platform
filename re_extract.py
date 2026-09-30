"""Backward-compatible entry point for frame re-extraction."""

from pipeline.frames.re_extract import *  # noqa: F401,F403
from pipeline.frames.re_extract import main


if __name__ == "__main__":
    raise SystemExit(main())
