#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command_args=("$@")
hardware_command="${1:-}"
network_interface=""
domain_id="0"
for ((index = 0; index < ${#command_args[@]}; index++)); do
    case "${command_args[index]}" in
        --network-interface)
            if ((index + 1 < ${#command_args[@]})); then
                network_interface="${command_args[index + 1]}"
            fi
            ;;
        --network-interface=*) network_interface="${command_args[index]#*=}" ;;
        --domain-id)
            if ((index + 1 < ${#command_args[@]})); then
                domain_id="${command_args[index + 1]}"
            fi
            ;;
        --domain-id=*) domain_id="${command_args[index]#*=}" ;;
    esac
done

if [[ -n "${G1_TABLETOP_ROS_PREFIX:-}" ]]; then
    ros_prefix="${G1_TABLETOP_ROS_PREFIX}"
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ros_prefix="/opt/ros/jazzy"
else
    ros_prefix="/opt/ros/humble"
fi
cyclone_prefix="${workspace_root}/deps/cyclonedds_python_prefix"
local_mcap_prefix="${workspace_root}/deps/rosbag2_mcap_prefix${ros_prefix}"
unitree_ros_setup="${G1_TABLETOP_UNITREE_ROS_SETUP:-${workspace_root}/../g1pilot_ws/install/setup.bash}"

if [[ ! -f "${ros_prefix}/setup.bash" || ! -d "${cyclone_prefix}/lib" ]]; then
    echo "hardware environment is unavailable; retain/install the commissioned account-local CycloneDDS runtime" >&2
    exit 1
fi
if [[ ! -x "${workspace_root}/.venv/bin/g1-tabletop" ]]; then
    echo "control environment is unavailable; run ./tools/setup_control_env.sh" >&2
    exit 1
fi
if [[ -z "${network_interface}" || ! "${network_interface}" =~ ^[[:alnum:]_.:-]+$ ]] || \
   [[ ! -e "/sys/class/net/${network_interface}" ]]; then
    echo "a valid --network-interface is required" >&2
    exit 1
fi
if [[ ! "${domain_id}" =~ ^[0-9]+$ ]]; then
    echo "invalid ROS domain ID: ${domain_id}" >&2
    exit 1
fi

set +u
source "${ros_prefix}/setup.bash"
if [[ "${hardware_command}" == "run-tabletop" || \
      "${hardware_command}" == "measure-seat-compliance" ]]; then
    if [[ ! -f "${unitree_ros_setup}" ]]; then
        echo "official Unitree ROS message installation is unavailable: ${unitree_ros_setup}" >&2
        exit 1
    fi
    source "${unitree_ros_setup}"
fi
if [[ "${hardware_command}" == "run-tabletop" || \
      "${hardware_command}" == "measure-seat-compliance" ]]; then
    if [[ ! -f "${local_mcap_prefix}/lib/librosbag2_storage_mcap.so" ]]; then
        echo "account-local MCAP plugin is unavailable; run ./tools/setup_recording_benchmark.sh once" >&2
        exit 1
    fi
fi
set -u
export CYCLONEDDS_HOME="${cyclone_prefix}"
export AMENT_PREFIX_PATH="${local_mcap_prefix}:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="${local_mcap_prefix}:${cyclone_prefix}:${ros_prefix}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${local_mcap_prefix}/lib:${cyclone_prefix}/lib:${ros_prefix}/lib:${ros_prefix}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${cyclone_prefix}/lib:${LIBRARY_PATH:-}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${domain_id}"
export ROS_LOCALHOST_ONLY=0
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
    export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"${network_interface}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

if [[ "${hardware_command}" == "run-tabletop" || \
      "${hardware_command}" == "measure-seat-compliance" ]]; then
    storage_plugins="$(ros2 bag list storage)"
    if ! grep -qx mcap <<<"${storage_plugins}"; then
        echo "account-local MCAP storage plugin was not discovered" >&2
        exit 1
    fi
fi
if [[ "${hardware_command}" == "run-tabletop" || \
      "${hardware_command}" == "measure-seat-compliance" ]]; then
    for message_type in \
        unitree_hg/msg/LowState \
        unitree_hg/msg/IMUState \
        unitree_hg/msg/LowCmd \
        unitree_hg/msg/HandState \
        unitree_hg/msg/HandCmd
    do
        if ! ros2 interface show "${message_type}" >/dev/null 2>&1; then
            echo "official Unitree ROS type support is unavailable: ${message_type}" >&2
            exit 1
        fi
    done
fi

cd "${workspace_root}"
case "${hardware_command}" in
    collect-calibration|run-tabletop|measure-seat-compliance)
        "${workspace_root}/tools/g1_realsense_pc2.sh" stop
        "${workspace_root}/tools/g1_realsense_pc2.sh" start
        ;;
esac
exec "${workspace_root}/.venv/bin/g1-tabletop" "$@"
