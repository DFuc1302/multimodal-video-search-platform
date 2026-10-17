"""Backward-compatible entry point for top OCR reruns."""

from pipeline.ocr.rerun_top import *  # noqa: F401,F403
from pipeline.ocr.rerun_top import main


if __name__ == "__main__":
    raise SystemExit(main())
