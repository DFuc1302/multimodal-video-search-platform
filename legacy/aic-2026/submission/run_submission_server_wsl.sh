#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TEAM_VENV="${TEAM_VENV:-/home/dtp/project/aic_local_prep/.venv_sys}"
source "$TEAM_VENV/bin/activate"
set -a
source .dres_credentials.env
set +a
export DATA_ROOT="${DATA_ROOT:-/mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index}"
export MAPFILES_DIR="${MAPFILES_DIR:-$DATA_ROOT/map-keyframes}"
export DRES_OFFICIAL_MAPS="${DRES_OFFICIAL_MAPS:-$DATA_ROOT/aic26-b2-map-keyframes.zip}"
exec python -m uvicorn submission_server:app --host "${SERVER_HOST:-localhost}" --port "${SERVER_PORT:-8080}" --no-access-log
