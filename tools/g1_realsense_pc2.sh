#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/g1_realsense_pc2.sh {start|status|stop}

Manage the temporary color-and-motion RealSense ROS node on G1 PC2.

  start   Stop the factory front-camera owner and launch RealSense ROS.
  status  Show factory and temporary RealSense process state.
  stop    Stop only the tracked RealSense node and restore the factory owner.

Environment overrides:
  G1_PC2_HOST          SSH destination (default: unitree@192.168.123.164)
  G1_PC2_SSH_IDENTITY  SSH private key path
EOF
}

action="${1:-}"
if [[ $# -ne 1 || ! "${action}" =~ ^(start|status|stop)$ ]]; then
    usage >&2
    exit 2
fi

pc2_host="${G1_PC2_HOST:-unitree@192.168.123.164}"
identity="${G1_PC2_SSH_IDENTITY:-${HOME}/.ssh/g1_pc2_ed25519}"
if [[ ! -f "${identity}" ]]; then
    echo "SSH identity not found: ${identity}" >&2
    exit 2
fi

ssh_options=(
    -i "${identity}"
    -o BatchMode=yes
    -o ConnectTimeout=5
)

ssh "${ssh_options[@]}" "${pc2_host}" bash -s -- "${action}" <<'REMOTE'
set -euo pipefail

action="$1"
factory_service="video_hub_pc4"
pid_file="/tmp/g1-calibration-realsense.pid"
log_file="/tmp/g1-calibration-realsense.log"
lock_file="/tmp/g1-calibration-realsense.lock"
camera_serial="348522074178"
camera_profile="1280x720x15"
depth_profile="640x480x15"

exec 9>"${lock_file}"
if ! flock -w 10 9; then
    echo "timed out waiting for the PC2 camera lifecycle lock" >&2
    exit 3
fi

factory_running() {
    /unitree/sbin/mscli getservice "${factory_service}" 2>/dev/null \
        | grep -q 'status:0'
}

wait_for_factory_state() {
    local expected="$1"
    local attempt
    for attempt in $(seq 1 100); do
        if [[ "${expected}" == "running" ]] && factory_running; then
            return 0
        fi
        if [[ "${expected}" == "stopped" ]] && ! factory_running; then
            return 0
        fi
        sleep 0.1
    done
    echo "factory front-camera service did not become ${expected}" >&2
    return 1
}

stop_factory() {
    if factory_running; then
        # This firmware returns status 1 even when the stop succeeds. Verify the
        # observed service state instead of trusting the command exit code.
        /unitree/sbin/mscli stopservice "${factory_service}" || true
    fi
    wait_for_factory_state stopped
}

start_factory() {
    if ! factory_running; then
        /unitree/sbin/mscli startservice "${factory_service}" || true
    fi
    wait_for_factory_state running
}

tracked_pid() {
    if [[ ! -f "${pid_file}" ]]; then
        return 1
    fi
    local value
    value=$(sed -n '1p' "${pid_file}")
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "invalid tracked RealSense PID: ${value}" >&2
        return 2
    fi
    printf '%s\n' "${value}"
}

valid_launcher() {
    local pid="$1"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    local command
    command=$(tr '\0' ' ' < "/proc/${pid}/cmdline")
    [[ "${command}" == *"/opt/ros/foxy/bin/ros2 launch realsense2_camera rs_launch.py"* ]]
}

expected_launcher() {
    local pid="$1"
    valid_launcher "${pid}" || return 1
    local command
    command=$(tr '\0' ' ' < "/proc/${pid}/cmdline")
    [[ "${command}" == *"serial_no:=_${camera_serial}"* ]]
    [[ "${command}" == *"rgb_camera.profile:=${camera_profile}"* ]]
    [[ "${command}" == *"enable_depth:=true"* ]]
    [[ "${command}" == *"depth_module.profile:=${depth_profile}"* ]]
    [[ "${command}" == *"enable_gyro:=true"* ]]
    [[ "${command}" == *"enable_accel:=true"* ]]
    [[ "${command}" == *"unite_imu_method:=0"* ]]
}

tracked_node_running() {
    local pid
    pid=$(tracked_pid 2>/dev/null) || return 1
    kill -0 "${pid}" 2>/dev/null && valid_launcher "${pid}"
}

stop_tracked_node() {
    local pid
    if ! pid=$(tracked_pid); then
        rm -f "${pid_file}"
        return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "${pid_file}"
        return 0
    fi
    if ! valid_launcher "${pid}"; then
        echo "refusing to stop unexpected tracked PID ${pid}" >&2
        return 4
    fi

    local children=()
    mapfile -t children < <(pgrep -P "${pid}" || true)
    local child command
    for child in "${children[@]}"; do
        [[ -r "/proc/${child}/cmdline" ]] || continue
        command=$(tr '\0' ' ' < "/proc/${child}/cmdline")
        if [[ "${command}" != *"/opt/ros/foxy/lib/realsense2_camera/realsense2_camera_node"* ]]; then
            echo "refusing to stop unexpected child PID ${child}: ${command}" >&2
            return 4
        fi
    done

    if ((${#children[@]})); then
        kill -TERM "${children[@]}" 2>/dev/null || true
    fi
    kill -TERM "${pid}" 2>/dev/null || true

    local attempt alive
    for attempt in $(seq 1 100); do
        alive=0
        kill -0 "${pid}" 2>/dev/null && alive=1
        for child in "${children[@]}"; do
            kill -0 "${child}" 2>/dev/null && alive=1
        done
        ((alive == 0)) && break
        sleep 0.1
    done

    if kill -0 "${pid}" 2>/dev/null; then
        echo "tracked RealSense launcher did not stop: ${pid}" >&2
        return 5
    fi
    for child in "${children[@]}"; do
        if kill -0 "${child}" 2>/dev/null; then
            echo "tracked RealSense child did not stop: ${child}" >&2
            return 5
        fi
    done
    rm -f "${pid_file}"
}

show_status() {
    echo "Factory front-camera service:"
    /unitree/sbin/mscli getservice "${factory_service}" || true
    /unitree/sbin/mscli getenable "${factory_service}" || true
    echo "Temporary RealSense ROS node:"
    local pid
    if pid=$(tracked_pid 2>/dev/null) && kill -0 "${pid}" 2>/dev/null; then
        if valid_launcher "${pid}"; then
            ps -o pid,ppid,stat,etime,cmd -p "${pid}" --ppid "${pid}"
            grep 'Open profile:' "${log_file}" 2>/dev/null | tail -n 6 || true
        else
            echo "tracked PID ${pid} is not the expected ROS launch process"
        fi
    else
        echo "not running"
    fi
}

case "${action}" in
    status)
        show_status
        ;;
    stop)
        stop_tracked_node
        start_factory
        show_status
        ;;
    start)
        if tracked_node_running; then
            pid=$(tracked_pid)
            if ! expected_launcher "${pid}"; then
                echo "tracked RealSense node uses an unexpected serial or profile" >&2
                exit 6
            fi
            stop_factory
            echo "RealSense ROS node is already running with the expected profile"
            show_status
            exit 0
        fi

        if [[ -f "${pid_file}" ]]; then
            stop_tracked_node
        fi
        if pgrep -f '^/opt/ros/foxy/lib/realsense2_camera/realsense2_camera_node' \
            >/dev/null; then
            echo "an untracked RealSense camera node is already running; refusing to replace it" >&2
            exit 6
        fi

        started=0
        rollback_required=1
        rollback_on_exit() {
            local status="$?"
            trap - EXIT INT TERM
            if ((rollback_required)); then
                set +e
                if ((started)); then
                    stop_tracked_node
                fi
                start_factory
                echo "RealSense startup failed; restored the factory front-camera service" >&2
            fi
            exit "${status}"
        }
        trap rollback_on_exit EXIT
        trap 'exit 130' INT TERM

        stop_factory

        required_setups=(
            /opt/ros/foxy/setup.bash
            /home/unitree/cyclonedds_ws/install/setup.bash
            /home/unitree/unitree_ros2/install/setup.bash
            /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
        )
        for setup_file in "${required_setups[@]}"; do
            if [[ ! -f "${setup_file}" ]]; then
                echo "missing PC2 ROS setup: ${setup_file}" >&2
                false
            fi
        done

        set +u
        source /opt/ros/foxy/setup.bash
        source /home/unitree/cyclonedds_ws/install/setup.bash
        source /home/unitree/unitree_ros2/install/setup.bash
        source /home/unitree/unitree_ros2/cyclonedds_ws/install/setup.bash
        set -u
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        export ROS_DOMAIN_ID=0
        export ROS_LOCALHOST_ONLY=0
        export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml

        : > "${log_file}"
        nohup setsid ros2 launch realsense2_camera rs_launch.py \
            camera_name:=camera \
            serial_no:=_348522074178 \
            enable_color:=true \
            rgb_camera.profile:=1280x720x15 \
            enable_depth:=true \
            depth_module.profile:=640x480x15 \
            enable_infra1:=false \
            enable_infra2:=false \
            enable_fisheye1:=false \
            enable_fisheye2:=false \
            enable_confidence:=false \
            enable_gyro:=true \
            enable_accel:=true \
            unite_imu_method:=0 \
            enable_pose:=false \
            pointcloud.enable:=false \
            align_depth.enable:=false \
            colorizer.enable:=false \
            enable_sync:=false \
            initial_reset:=false \
            >"${log_file}" 2>&1 </dev/null 9>&- &
        pid=$!
        printf '%s\n' "${pid}" > "${pid_file}"
        started=1

        for attempt in $(seq 1 150); do
            grep -q 'RealSense Node Is Up!' "${log_file}" && break
            kill -0 "${pid}" 2>/dev/null || false
            sleep 0.1
        done
        grep -q "Device Serial No: ${camera_serial}" "${log_file}"
        grep -q 'Device USB type: 3.2' "${log_file}"
        grep -q 'Open profile:.*Format: RGB8, Width: 1280, Height: 720, FPS: 15' \
            "${log_file}"
        grep -q 'Open profile:.*stream_type: Depth.*Width: 640, Height: 480, FPS: 15' \
            "${log_file}"
        grep -q 'Open profile:.*stream_type: Gyro' "${log_file}"
        grep -q 'Open profile:.*stream_type: Accel' "${log_file}"
        grep -q 'RealSense Node Is Up!' "${log_file}"
        expected_launcher "${pid}"
        pgrep -P "${pid}" -f \
            '/opt/ros/foxy/lib/realsense2_camera/realsense2_camera_node' \
            >/dev/null

        rollback_required=0
        trap - EXIT INT TERM
        echo "started RealSense ROS for D435i ${camera_serial}: ${camera_profile} RGB8, ${depth_profile} Z16, raw gyro and accelerometer"
        show_status
        ;;
esac
REMOTE
