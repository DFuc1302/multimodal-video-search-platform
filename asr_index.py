"""Backward-compatible entry point for ASR indexing."""

from pipeline.asr.index import *  # noqa: F401,F403
from pipeline.asr.index import main


if __name__ == "__main__":
    raise SystemExit(main())
