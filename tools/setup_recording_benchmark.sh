#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ros_distro="${ROS_DISTRO:-humble}"
local_root="${workspace_root}/deps/rosbag2_mcap_prefix"
local_prefix="${local_root}/opt/ros/${ros_distro}"

if [[ -f "${local_prefix}/lib/librosbag2_storage_mcap.so" ]]; then
    echo "account-local ROS ${ros_distro} MCAP plugin is already available"
    exit 0
fi

package_dir="$(mktemp -d /tmp/g1-rosbag2-mcap.XXXXXX)"
trap 'rm -rf "${package_dir}"' EXIT
mkdir -p "${local_root}"
cd "${package_dir}"
apt download \
    "ros-${ros_distro}-mcap-vendor" \
    "ros-${ros_distro}-rosbag2-storage-mcap"
for package in ./*.deb; do
    dpkg-deb -x "${package}" "${local_root}"
done

if [[ ! -f "${local_prefix}/lib/librosbag2_storage_mcap.so" ]]; then
    echo "account-local MCAP extraction did not produce the storage plugin" >&2
    exit 1
fi
echo "installed account-local ROS ${ros_distro} MCAP plugin under ${local_prefix}"
