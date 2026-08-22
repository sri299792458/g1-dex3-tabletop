#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spark_root="${G1_TABLETOP_SPARK_ROOT:-/home/kanth042/spark-data-collection}"
python_bin="${spark_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
    echo "conversion environment is unavailable; run ./tools/setup_lerobot_conversion.sh" >&2
    exit 1
fi

export PYTHONPATH="${workspace_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" -m g1_dex3_tabletop.lerobot_conversion "$@"
