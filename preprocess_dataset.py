"""Backward-compatible entry point for the preprocessing pipeline."""

from pipeline.preprocess.run import *  # noqa: F401,F403
from pipeline.preprocess.run import main


if __name__ == "__main__":
    raise SystemExit(main())
