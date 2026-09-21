#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 left|right [collect-calibration options]" >&2
    exit 2
fi

arm="$1"
shift
if [[ "${arm}" != "left" && "${arm}" != "right" ]]; then
    echo "first argument must be left or right" >&2
    exit 2
fi

exec "${workspace_root}/tools/g1_tabletop_hardware.sh" \
    collect-calibration \
    --arm "${arm}" \
    "$@"
