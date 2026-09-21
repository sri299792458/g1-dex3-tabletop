#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_prefix="${G1_CALIB_ROS_PREFIX:-/opt/ros/humble}"
ceres_prefix="${workspace_root}/deps/ceres_prefix"
local_root="${workspace_root}/deps/robot_calibration_prefix"
local_ros="${local_root}${ros_prefix}"

for required in \
    "${ros_prefix}/setup.bash" \
    "${ceres_prefix}/lib/libceres.so" \
    "${workspace_root}/install/setup.bash"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing account-local robot_calibration dependency: ${required}" >&2
        echo "run ./tools/install_robot_calibration_local.sh first" >&2
        exit 1
    fi
done

set +u
source "${ros_prefix}/setup.bash"
set -u
export CMAKE_PREFIX_PATH="${ceres_prefix}:${local_ros}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${ceres_prefix}/lib:${local_ros}/lib:${local_root}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
set +u
source "${workspace_root}/install/setup.bash"
set -u

exec "$@"
