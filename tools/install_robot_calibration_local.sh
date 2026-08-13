#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_prefix="${G1_CALIB_ROS_PREFIX:-/opt/ros/humble}"
ceres_source="${workspace_root}/deps/ceres-solver"
ceres_build="${workspace_root}/deps/ceres-build"
ceres_prefix="${workspace_root}/deps/ceres_prefix"
local_root="${workspace_root}/deps/robot_calibration_prefix"
local_ros="${local_root}${ros_prefix}"
robot_calibration_source="${workspace_root}/third_party/robot_calibration"
robot_calibration_revision="db991b040d1dc28af09d8865fc72f09720e12b73"
ceres_revision="399cda773035d99eaf1f4a129a666b3c4df9d1b1"

if [[ ! -f "${ros_prefix}/setup.bash" ]]; then
    echo "ROS setup not found: ${ros_prefix}/setup.bash" >&2
    exit 1
fi
if [[ "$(git -C "${robot_calibration_source}" rev-parse HEAD)" != \
      "${robot_calibration_revision}" ]]; then
    echo "robot_calibration submodule revision verification failed" >&2
    exit 1
fi

if [[ ! -d "${ceres_source}/.git" ]]; then
    git clone https://github.com/ceres-solver/ceres-solver.git "${ceres_source}"
fi
git -C "${ceres_source}" fetch --quiet origin "${ceres_revision}"
git -C "${ceres_source}" checkout --quiet --detach "${ceres_revision}"
if [[ "$(git -C "${ceres_source}" rev-parse HEAD)" != "${ceres_revision}" ]]; then
    echo "Ceres revision verification failed" >&2
    exit 1
fi

cmake -S "${ceres_source}" -B "${ceres_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ceres_prefix}" \
    -DMINIGLOG=ON \
    -DGFLAGS=OFF \
    -DSUITESPARSE=OFF \
    -DCXSPARSE=OFF \
    -DLAPACK=OFF \
    -DEIGENSPARSE=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_SHARED_LIBS=ON
cmake --build "${ceres_build}" --parallel "${G1_CALIB_BUILD_JOBS:-4}"
cmake --install "${ceres_build}"

download_directory="$(mktemp -d)"
cleanup() {
    rm -rf -- "${download_directory}"
}
trap cleanup EXIT
pushd "${download_directory}" >/dev/null
apt download \
    libccd2 \
    libfcl0.7 \
    liboctomap1.9 \
    ros-humble-camera-calibration-parsers \
    ros-humble-control-msgs \
    ros-humble-cv-bridge \
    ros-humble-eigen-stl-containers \
    ros-humble-geometric-shapes \
    ros-humble-moveit-msgs \
    ros-humble-object-recognition-msgs \
    ros-humble-octomap-msgs \
    ros-humble-random-numbers
mkdir -p "${local_root}"
for package in ./*.deb; do
    dpkg-deb -x "${package}" "${local_root}"
done
popd >/dev/null

ln -sfn cv_bridge.h \
    "${local_ros}/include/cv_bridge/cv_bridge/cv_bridge.hpp"

set +u
source "${ros_prefix}/setup.bash"
set -u
export CMAKE_PREFIX_PATH="${ceres_prefix}:${local_ros}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${ceres_prefix}/lib:${local_ros}/lib:${local_root}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

cd "${workspace_root}"
colcon build \
    --base-paths third_party/robot_calibration \
    --packages-select robot_calibration_msgs
set +u
source "${workspace_root}/install/robot_calibration_msgs/share/robot_calibration_msgs/local_setup.bash"
set -u
colcon build \
    --base-paths third_party/robot_calibration \
    --packages-select robot_calibration \
    --cmake-clean-cache \
    --cmake-args \
        -DBUILD_TESTING=OFF \
        -DCeres_DIR="${ceres_prefix}/lib/cmake/Ceres" \
        -Dcv_bridge_DIR="${local_ros}/share/cv_bridge/cmake" \
        -Dcamera_calibration_parsers_DIR="${local_ros}/share/camera_calibration_parsers/cmake" \
        -Dcontrol_msgs_DIR="${local_ros}/share/control_msgs/cmake" \
        -Dgeometric_shapes_DIR="${local_ros}/share/geometric_shapes/cmake" \
        -Dmoveit_msgs_DIR="${local_ros}/share/moveit_msgs/cmake"

echo "account-local robot_calibration is ready"
