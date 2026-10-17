"""Backward-compatible entry point for Redis migration."""

from pipeline.index.redis import *  # noqa: F401,F403
from pipeline.index.redis import main


if __name__ == "__main__":
    raise SystemExit(main())
