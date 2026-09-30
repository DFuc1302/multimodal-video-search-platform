"""Backward-compatible entry point for the preprocessing pipeline."""

import sys

from pipeline.preprocess.run import *  # noqa: F401,F403
from pipeline.preprocess.run import main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
