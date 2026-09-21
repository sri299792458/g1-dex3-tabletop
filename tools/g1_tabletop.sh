#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "solve-calibration" ]]; then
    exec "${workspace_root}/tools/g1_robot_calibration.sh" \
        "${workspace_root}/.venv/bin/g1-tabletop" "$@"
fi
exec "${workspace_root}/.venv/bin/g1-tabletop" "$@"
