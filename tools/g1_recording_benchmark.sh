#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_prefix="${G1_TABLETOP_ROS_PREFIX:-/opt/ros/humble}"
if [[ ! -f "${ros_prefix}/setup.bash" ]]; then
    echo "ROS setup not found under ${ros_prefix}" >&2
    exit 1
fi
if [[ ! -x "${workspace_root}/.venv/bin/python" ]]; then
    echo "control environment is unavailable; run ./tools/setup_control_env.sh" >&2
    exit 1
fi

set +u
source "${ros_prefix}/setup.bash"
set -u
"${workspace_root}/tools/setup_recording_benchmark.sh"

local_mcap_prefix="${workspace_root}/deps/rosbag2_mcap_prefix${ros_prefix}"
export AMENT_PREFIX_PATH="${local_mcap_prefix}:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="${local_mcap_prefix}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${local_mcap_prefix}/lib:${LD_LIBRARY_PATH:-}"
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${G1_RECORDING_BENCHMARK_DOMAIN_ID:-221}"
unset CYCLONEDDS_URI

if ! ros2 bag list storage | grep -qx mcap; then
    echo "account-local MCAP storage plugin was not discovered" >&2
    exit 1
fi

cd "${workspace_root}"
exec "${workspace_root}/.venv/bin/python" -m g1_dex3_tabletop.recording_benchmark run "$@"
