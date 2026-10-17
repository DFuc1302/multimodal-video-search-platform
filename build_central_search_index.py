"""Backward-compatible entry point for central index construction."""

from pipeline.index.central import *  # noqa: F401,F403
from pipeline.index.central import main


if __name__ == "__main__":
    raise SystemExit(main())
