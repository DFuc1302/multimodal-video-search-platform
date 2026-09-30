"""Backward-compatible entry point for OCR reindexing."""

from pipeline.ocr.reindex import *  # noqa: F401,F403
from pipeline.ocr.reindex import main


if __name__ == "__main__":
    raise SystemExit(main())
