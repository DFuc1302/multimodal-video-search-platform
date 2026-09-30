"""Backward-compatible entry point for shard export."""

from pipeline.index.export import *  # noqa: F401,F403
from pipeline.index.export import main


if __name__ == "__main__":
    raise SystemExit(main())
