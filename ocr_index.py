"""Backward-compatible entry point for OCR indexing."""

from pipeline.ocr.index import *  # noqa: F401,F403
from pipeline.ocr.index import main


if __name__ == "__main__":
    raise SystemExit(main())
