#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_prefix="${G1_TABLETOP_ROS_PREFIX:-/opt/ros/humble}"
unitree_ros_setup="${G1_TABLETOP_UNITREE_ROS_SETUP:-${workspace_root}/../g1pilot_ws/install/setup.bash}"
local_mcap_prefix="${workspace_root}/deps/rosbag2_mcap_prefix${ros_prefix}"

if [[ ! -f "${ros_prefix}/setup.bash" ]]; then
    echo "ROS setup not found under ${ros_prefix}" >&2
    exit 1
fi
if [[ ! -f "${unitree_ros_setup}" ]]; then
    echo "official Unitree ROS message installation is unavailable: ${unitree_ros_setup}" >&2
    exit 1
fi
if [[ ! -x "${workspace_root}/.venv/bin/python" ]]; then
    echo "control environment is unavailable; run ./tools/setup_control_env.sh" >&2
    exit 1
fi
if [[ ! -f "${local_mcap_prefix}/lib/librosbag2_storage_mcap.so" ]]; then
    echo "account-local MCAP plugin is unavailable; run ./tools/setup_recording_benchmark.sh once" >&2
    exit 1
fi

set +u
source "${ros_prefix}/setup.bash"
source "${unitree_ros_setup}"
set -u
export AMENT_PREFIX_PATH="${local_mcap_prefix}:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="${local_mcap_prefix}:${ros_prefix}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${local_mcap_prefix}/lib:${ros_prefix}/lib:${ros_prefix}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

cd "${workspace_root}"
module="g1_dex3_tabletop.state_estimation_replay"
if [[ "${1:-}" == "continuous" ]]; then
    module="g1_dex3_tabletop.continuous_state_estimation_replay"
    shift
elif [[ "${1:-}" == "depth-plane" ]]; then
    module="g1_dex3_tabletop.depth_plane_replay"
    shift
fi
exec "${workspace_root}/.venv/bin/python" -m "${module}" "$@"
