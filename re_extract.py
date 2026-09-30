"""Backward-compatible entry point for frame re-extraction."""

import sys

from pipeline.frames.re_extract import *  # noqa: F401,F403
from pipeline.frames.re_extract import main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
