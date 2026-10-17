"""Backward-compatible entry point for ASR status checks."""

from pipeline.asr.status import *  # noqa: F401,F403
from pipeline.asr.status import main


if __name__ == "__main__":
    raise SystemExit(main())
