#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spark_root="${G1_TABLETOP_SPARK_ROOT:-/home/kanth042/spark-data-collection}"
lerobot_root="${G1_TABLETOP_LEROBOT_ROOT:-/home/kanth042/lerobot}"
venv="${spark_root}/.venv"
expected_lerobot_revision="7e241bd630a3719a56157a497ce5d08f244784f1"

if [[ ! -d "${spark_root}/.git" ]]; then
    echo "missing SPARK checkout at ${spark_root}" >&2
    exit 1
fi
if [[ ! -d "${lerobot_root}/.git" ]]; then
    echo "missing LeRobot checkout at ${lerobot_root}" >&2
    exit 1
fi
actual_revision="$(git -C "${lerobot_root}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${expected_lerobot_revision}" ]]; then
    echo "LeRobot must be at tested v0.6.1 revision ${expected_lerobot_revision}; got ${actual_revision}" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create the account-local Python 3.12 converter environment" >&2
    exit 1
fi

uv venv --python 3.12 "${venv}"
uv pip install --python "${venv}/bin/python" \
    'torch==2.7.1' 'torchvision==0.22.1' \
    --default-index https://download.pytorch.org/whl/cpu
uv pip install --python "${venv}/bin/python" \
    -r "${spark_root}/data_pipeline/requirements-converter.txt" \
    mcap-ros2-support pyyaml
uv pip install --python "${venv}/bin/python" --no-deps -e "${lerobot_root}"

echo "LeRobot conversion environment ready at ${venv}"
echo "This environment is offline-only and does not alter the G1 control environments."
