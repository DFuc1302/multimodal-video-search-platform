"""Emit shell-quoted environment defaults from a verified active dataset file."""
import argparse
import json
import os
import shlex
from pathlib import Path

ALLOWED = {
    "DATA_ROOT", "KEYFRAMES_DIR", "MAPFILES_DIR", "VIDEOS_DIR", "BACKEND",
    "REDIS_PREFIX", "REDIS_URL", "REDIS_PORT", "REDIS_ROOT", "REDIS_DATA_DIR",
    "REQUIRE_REDIS_SOURCE_MATCH", "REDIS_MAXMEMORY", "SUMMARY_BACKEND", "SUMMARY_INDEX_DIR",
}


def exports(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported active dataset configuration")
    values = payload["environment"]
    if not isinstance(values, dict) or set(values) - ALLOWED:
        raise ValueError("Invalid active dataset environment")
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in values.items()
                     if not os.getenv(key))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    print(exports(parser.parse_args().config))
