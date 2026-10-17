"""Backward-compatible entry point for processed-data merging."""

from pipeline.preprocess.merge import *  # noqa: F401,F403
from pipeline.preprocess.merge import main


if __name__ == "__main__":
    raise SystemExit(main())
