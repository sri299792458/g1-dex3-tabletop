#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${workspace_root}/generated/tabletop_cube"

if [[ ! -x "${workspace_root}/.venv/bin/aprilcube" ]]; then
    echo "control environment is unavailable; run ./tools/setup_control_env.sh" >&2
    exit 1
fi

mkdir -p "${workspace_root}/generated"
if [[ -e "${output}" ]]; then
    echo "generated tabletop cube already exists: ${output}"
    exit 0
fi

exec "${workspace_root}/.venv/bin/aprilcube" generate \
    "${workspace_root}/config/tabletop/cube_head.yaml" \
    -o "${output}"
