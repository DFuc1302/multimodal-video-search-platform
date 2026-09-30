"""Backward-compatible entry point for the reusable pipeline runner."""

from pipeline.run import *  # noqa: F401,F403
from pipeline.run import main


if __name__ == "__main__":
    raise SystemExit(main())
