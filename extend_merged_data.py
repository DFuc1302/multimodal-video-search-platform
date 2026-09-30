"""Backward-compatible entry point for extending processed data."""

from pipeline.preprocess.extend import *  # noqa: F401,F403
from pipeline.preprocess.extend import main


if __name__ == "__main__":
    raise SystemExit(main())
