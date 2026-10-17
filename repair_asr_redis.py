"""Backward-compatible entry point for ASR Redis repair."""

from pipeline.asr.repair_redis import *  # noqa: F401,F403
from pipeline.asr.repair_redis import main


if __name__ == "__main__":
    raise SystemExit(main())
