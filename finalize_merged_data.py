"""Backward-compatible entry point for finalizing merged data."""

from pipeline.preprocess.finalize import *  # noqa: F401,F403
from pipeline.preprocess.finalize import main


if __name__ == "__main__":
    raise SystemExit(main())
