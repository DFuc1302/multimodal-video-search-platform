"""Backward-compatible entry point for the new-dataset pipeline."""

from pipeline.new_dataset import *  # noqa: F401,F403
from pipeline.new_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
