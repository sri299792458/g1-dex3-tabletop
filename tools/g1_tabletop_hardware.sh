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
set -u
export CYCLONEDDS_HOME="${cyclone_prefix}"
export CMAKE_PREFIX_PATH="${cyclone_prefix}:${ros_prefix}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${cyclone_prefix}/lib:${ros_prefix}/lib:${ros_prefix}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${cyclone_prefix}/lib:${LIBRARY_PATH:-}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${domain_id}"
export ROS_LOCALHOST_ONLY=0
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
    export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"${network_interface}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

cd "${workspace_root}"
case "${hardware_command}" in
    collect-calibration|run-tabletop)
        "${workspace_root}/tools/g1_realsense_pc2.sh" stop
        "${workspace_root}/tools/g1_realsense_pc2.sh" start
        ;;
esac
exec "${workspace_root}/.venv/bin/g1-tabletop" "$@"
