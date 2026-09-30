"""Backward-compatible entry point for frame deduplication."""

from pipeline.frames.dedup import *  # noqa: F401,F403
from pipeline.frames.dedup import main


if __name__ == "__main__":
    raise SystemExit(main())
