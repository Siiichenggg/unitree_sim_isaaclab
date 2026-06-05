#!/usr/bin/env python3

"""Minimal launcher for OR VLM navigation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STATE_DIR = PROJECT_ROOT / ".tmp" / "or_vlm_nav"
LOG_DIR = PROJECT_ROOT / "logs" / "or_vlm_nav"
PID_FILE = STATE_DIR / "sim.pid"
META_FILE = STATE_DIR / "sim.json"
SIM_LOG = LOG_DIR / "sim.log"
OR_AGENT = PROJECT_ROOT / "tools" / "or_agent_graph.py"
WASD_TELEOP = PROJECT_ROOT / "tools" / "wasd_walk_teleop.py"
SIM_MAIN = PROJECT_ROOT / "sim_main.py"
AGENT_PYTHON = Path.home() / "miniconda3" / "envs" / "or_agent_env" / "bin" / "python"
OR_SCENES = {
    "halo": PROJECT_ROOT / "assets" / "objects" / "OR" / "Model" / "halo_room_baked" / "halo_room_baked.usd",
    "pulm": PROJECT_ROOT / "assets" / "objects" / "OR" / "Model" / "pulm_room_baked" / "pulm_room_baked.usd",
}

TASK = "Isaac-OR-VLM-G129-Dex3-Wholebody"
HAND_FLAG = "--enable_dex3_dds"
DDS_CHANNEL = 0
DDS_INTERFACE = "lo"
SCENE_CAMERA_SENSOR_NAMES = ",".join(f"scene_camera_{index:02d}" for index in range(11))
CAMERA_INCLUDE = (
    f"front_camera,left_wrist_camera,right_wrist_camera,{SCENE_CAMERA_SENSOR_NAMES}"
)
CAMERA_READY_TIMEOUT = 90.0
OR_CAMERA_WIDTH = os.environ.get("UNITREE_OR_CAMERA_WIDTH", "1920")
OR_CAMERA_HEIGHT = os.environ.get("UNITREE_OR_CAMERA_HEIGHT", "1080")
OR_CAMERA_JPEG_QUALITY = os.environ.get("UNITREE_OR_CAMERA_JPEG_QUALITY", "95")
OR_CAMERA_TRANSPORT = os.environ.get("UNITREE_CAMERA_TRANSPORT", "dds")
OR_CAMERA_WRITE_INTERVAL = os.environ.get("UNITREE_OR_CAMERA_WRITE_INTERVAL", "4")
OR_WHOLEBODY_ACTION_SUBSTEPS = os.environ.get("UNITREE_WHOLEBODY_ACTION_SUBSTEPS", "8")
os.environ.setdefault("UNITREE_CAMERA_TRANSPORT", OR_CAMERA_TRANSPORT)
os.environ.setdefault("UNITREE_DDS_DOMAIN_ID", str(DDS_CHANNEL))
os.environ.setdefault("UNITREE_DDS_INTERFACE", DDS_INTERFACE)
os.environ.setdefault("CAMERA_JPEG_QUALITY", OR_CAMERA_JPEG_QUALITY)
AGENT_MAX_ENV_CAMERAS = os.environ.get("UNITREE_OR_AGENT_MAX_ENV_CAMERAS", "11")
AGENT_MAX_OBSTACLE_DETECTIONS = os.environ.get("UNITREE_OR_AGENT_MAX_OBSTACLE_DETECTIONS", "2")
AGENT_MAX_OBSTACLE_CAMERAS = os.environ.get("UNITREE_OR_AGENT_MAX_OBSTACLE_CAMERAS", "4")
AGENT_LOCAL_APPROACH_DURATION = os.environ.get("UNITREE_OR_AGENT_LOCAL_APPROACH_DURATION", "30")
AGENT_LOCAL_APPROACH_RETRY_LIMIT = os.environ.get("UNITREE_OR_AGENT_LOCAL_APPROACH_RETRY_LIMIT", "10")
AGENT_LOCAL_STOP_AREA_RATIO = os.environ.get("UNITREE_OR_AGENT_LOCAL_STOP_AREA_RATIO", "0.07")
AGENT_LOCAL_SLOW_AREA_RATIO = os.environ.get("UNITREE_OR_AGENT_LOCAL_SLOW_AREA_RATIO", "0.16")
AGENT_LOCAL_STOP_HEIGHT_RATIO = os.environ.get("UNITREE_OR_AGENT_LOCAL_STOP_HEIGHT_RATIO", "0.35")
AGENT_LOCAL_MAX_SPEED = os.environ.get("UNITREE_OR_AGENT_LOCAL_MAX_SPEED", "0.28")
AGENT_LOCAL_MIN_SPEED = os.environ.get("UNITREE_OR_AGENT_LOCAL_MIN_SPEED", "0.08")
AGENT_LOCAL_MAX_YAW_RATE = os.environ.get("UNITREE_OR_AGENT_LOCAL_MAX_YAW_RATE", "0.42")
AGENT_LOCAL_YAW_KP = os.environ.get("UNITREE_OR_AGENT_LOCAL_YAW_KP", "0.55")
AGENT_LOCAL_FORWARD_CENTER_LIMIT = os.environ.get("UNITREE_OR_AGENT_LOCAL_FORWARD_CENTER_LIMIT", "0.35")
AGENT_MAX_STEPS = os.environ.get("UNITREE_OR_AGENT_MAX_STEPS", "100")
AGENT_GOAL_TOLERANCE = os.environ.get("UNITREE_OR_AGENT_GOAL_TOLERANCE", "0.25")
AGENT_YAW_RATE = os.environ.get("UNITREE_OR_AGENT_YAW_RATE", "2.0")
AGENT_NAV_COMMAND_DURATION = os.environ.get("UNITREE_OR_AGENT_NAV_COMMAND_DURATION", "1.5")
AGENT_NAV_RETRY_LIMIT = os.environ.get("UNITREE_OR_AGENT_NAV_RETRY_LIMIT", "40")
AGENT_TARGET_ARRIVAL_RADIUS = os.environ.get("UNITREE_OR_AGENT_TARGET_ARRIVAL_RADIUS", "0.85")
AGENT_NEAR_GOAL_ACCEPTANCE = os.environ.get("UNITREE_OR_AGENT_NEAR_GOAL_ACCEPTANCE", "0.55")
AGENT_NEAR_GOAL_STALL_ATTEMPTS = os.environ.get("UNITREE_OR_AGENT_NEAR_GOAL_STALL_ATTEMPTS", "3")


def detect_sim_python() -> str:
    override = os.environ.get("UNITREE_SIM_PYTHON")
    if override:
        return override
    candidates = [
        Path.home() / "miniconda3" / "envs" / "unitree_sim_env" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "env_isaaclab" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def detect_agent_python() -> str:
    override = os.environ.get("UNITREE_OR_AGENT_PYTHON")
    if override:
        return override
    if AGENT_PYTHON.is_file():
        return str(AGENT_PYTHON)
    return detect_sim_python()


def sim_python_has_isaac(python_bin: str) -> bool:
    result = subprocess.run(
        [python_bin, "-c", "import isaaclab, isaacsim"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid() -> int | None:
    if not PID_FILE.is_file():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def read_metadata() -> dict:
    if not META_FILE.is_file():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_sim_command(python_bin: str) -> list[str]:
    return [
        python_bin,
        str(SIM_MAIN),
        "--device",
        "cpu",
        "--enable_cameras",
        "--task",
        TASK,
        "--robot_type",
        "g129",
        HAND_FLAG,
        "--action_source",
        "dds_wholebody",
        "--stats_interval",
        "10.0",
        "--disable_image_server",
        "--camera_include",
        CAMERA_INCLUDE,
        "--camera_jpeg_quality",
        OR_CAMERA_JPEG_QUALITY,
        "--camera_transport",
        OR_CAMERA_TRANSPORT,
        "--camera_write_interval",
        OR_CAMERA_WRITE_INTERVAL,
        "--wholebody_action_substeps",
        OR_WHOLEBODY_ACTION_SUBSTEPS,
    ]


def stop_simulator() -> int:
    pid = read_pid()
    if pid is None:
        return 0
    if not is_pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return 0

    print(f"[or-vlm] stopping simulator pid={pid}")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return 0

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            PID_FILE.unlink(missing_ok=True)
            print("[or-vlm] stopped")
            return 0
        time.sleep(0.2)

    print("[or-vlm] simulator did not exit after SIGTERM; sending SIGKILL")
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)
    return 0


def select_or_scene(default_scene: str = "halo") -> str | None:
    print("[or-vlm] select OR scene")
    scene_names = tuple(OR_SCENES)
    for index, scene in enumerate(scene_names, start=1):
        suffix = " default" if scene == default_scene else ""
        print(f"{index}) {scene}{suffix}")
    print("q) Cancel")

    while True:
        choice = input("scene> ").strip().lower()
        if not choice:
            return default_scene
        if choice in {"q", "quit", "exit"}:
            return None
        if choice in OR_SCENES:
            return choice
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(scene_names):
                return scene_names[index]
        print("[or-vlm] unknown scene")


def start_simulator(scene: str) -> int:
    ensure_dirs()
    if scene not in OR_SCENES:
        print(f"[or-vlm] unknown OR scene: {scene}")
        return 2

    python_bin = detect_sim_python()
    command = build_sim_command(python_bin)
    room_usd = str(OR_SCENES[scene])
    if not OR_SCENES[scene].is_file():
        print(f"[or-vlm] OR scene USD not found: {room_usd}")
        return 2

    pid = read_pid()
    if pid is not None and is_pid_alive(pid):
        metadata = read_metadata()
        same_runtime = (
            metadata.get("task") == TASK
            and metadata.get("room_usd") == room_usd
            and metadata.get("dds_channel") == DDS_CHANNEL
            and metadata.get("dds_interface") == DDS_INTERFACE
        )
        if same_runtime:
            print(f"[or-vlm] simulator already running pid={pid}")
            print(f"[or-vlm] log={SIM_LOG}")
            return 0
        stop_simulator()

    if not sim_python_has_isaac(python_bin):
        print(f"[or-vlm] simulator Python cannot import isaaclab/isaacsim: {python_bin}")
        print("[or-vlm] set UNITREE_SIM_PYTHON to the Isaac/IsaacLab Python if needed")
        return 2

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["UNITREE_OR_SCENE"] = scene
    env["UNITREE_ROOM_USD"] = room_usd
    env["UNITREE_DDS_DOMAIN_ID"] = str(DDS_CHANNEL)
    env["UNITREE_DDS_INTERFACE"] = DDS_INTERFACE
    env["UNITREE_OR_CAMERA_WIDTH"] = OR_CAMERA_WIDTH
    env["UNITREE_OR_CAMERA_HEIGHT"] = OR_CAMERA_HEIGHT
    env["CAMERA_JPEG_QUALITY"] = OR_CAMERA_JPEG_QUALITY
    env["UNITREE_CAMERA_TRANSPORT"] = OR_CAMERA_TRANSPORT
    env["UNITREE_OR_CAMERA_WRITE_INTERVAL"] = OR_CAMERA_WRITE_INTERVAL
    env["UNITREE_WHOLEBODY_ACTION_SUBSTEPS"] = OR_WHOLEBODY_ACTION_SUBSTEPS

    log_file = SIM_LOG.open("a", encoding="utf-8")
    log_file.write("\n\n=== start sim: " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
    log_file.write("command: " + " ".join(command) + "\n")
    log_file.write(f"room_usd: {room_usd}\n")
    log_file.flush()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    META_FILE.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "scene": scene,
                "task": TASK,
                "command": command,
                "room_usd": room_usd,
                "dds_channel": DDS_CHANNEL,
                "dds_interface": DDS_INTERFACE,
                "log": str(SIM_LOG),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[or-vlm] started simulator pid={process.pid}")
    print(f"[or-vlm] task={TASK}")
    print(f"[or-vlm] scene={scene}")
    print(f"[or-vlm] room_usd={room_usd}")
    print(f"[or-vlm] log={SIM_LOG}")
    return 0


def execute_structured_state_agent(instruction: str) -> int:
    instruction = instruction.strip()
    if not instruction:
        print("[or-vlm] missing instruction")
        return 2
    metadata = read_metadata()
    room_usd = metadata.get("room_usd")
    scene = metadata.get("scene", "halo")
    if not room_usd:
        print("[or-vlm] simulator metadata missing room_usd; start simulator first")
        return 2
    command = [
        detect_agent_python(),
        str(OR_AGENT),
        instruction,
        "--scene",
        scene,
        "--room-usd",
        room_usd,
        "--channel-id",
        str(DDS_CHANNEL),
        "--dds-interface",
        DDS_INTERFACE,
        "--sim-python",
        detect_sim_python(),
        "--max-env-cameras",
        AGENT_MAX_ENV_CAMERAS,
        "--max-obstacle-detections",
        AGENT_MAX_OBSTACLE_DETECTIONS,
        "--max-obstacle-cameras",
        AGENT_MAX_OBSTACLE_CAMERAS,
        "--local-approach-duration",
        AGENT_LOCAL_APPROACH_DURATION,
        "--local-approach-retry-limit",
        AGENT_LOCAL_APPROACH_RETRY_LIMIT,
        "--local-stop-area-ratio",
        AGENT_LOCAL_STOP_AREA_RATIO,
        "--local-slow-area-ratio",
        AGENT_LOCAL_SLOW_AREA_RATIO,
        "--local-stop-height-ratio",
        AGENT_LOCAL_STOP_HEIGHT_RATIO,
        "--local-max-speed",
        AGENT_LOCAL_MAX_SPEED,
        "--local-min-speed",
        AGENT_LOCAL_MIN_SPEED,
        "--local-max-yaw-rate",
        AGENT_LOCAL_MAX_YAW_RATE,
        "--local-yaw-kp",
        AGENT_LOCAL_YAW_KP,
        "--local-forward-center-limit",
        AGENT_LOCAL_FORWARD_CENTER_LIMIT,
        "--max-steps",
        AGENT_MAX_STEPS,
        "--goal-tolerance",
        AGENT_GOAL_TOLERANCE,
        "--yaw-rate",
        AGENT_YAW_RATE,
        "--navigation-command-duration",
        AGENT_NAV_COMMAND_DURATION,
        "--navigation-retry-limit",
        AGENT_NAV_RETRY_LIMIT,
        "--target-arrival-radius",
        AGENT_TARGET_ARRIVAL_RADIUS,
        "--near-goal-acceptance",
        AGENT_NEAR_GOAL_ACCEPTANCE,
        "--near-goal-stall-attempts",
        AGENT_NEAR_GOAL_STALL_ATTEMPTS,
    ]
    print(f"[or-vlm] structured-state instruction={instruction}", flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def _wait_for_dds_camera_images(camera_names: list[str], timeout_s: float) -> bool:
    script = r"""
import os
import sys
import time
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.shared_memory_utils import MultiImageReader

requested = [name for name in os.environ["UNITREE_OR_WAIT_CAMERAS"].split(",") if name]
timeout_s = float(os.environ.get("UNITREE_OR_CAMERA_READY_TIMEOUT", "90"))
deadline = time.monotonic() + timeout_s
reader = MultiImageReader()
try:
    while time.monotonic() < deadline:
        missing = []
        for name in requested:
            frame = reader.read_single_image(name)
            if frame is None:
                missing.append(name)
        if not missing:
            print("[camera-wait] ready " + ",".join(requested), flush=True)
            raise SystemExit(0)
        time.sleep(0.25)
    print("[camera-wait] timeout missing=" + ",".join(missing), flush=True)
    raise SystemExit(1)
finally:
    reader.close()
"""
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["UNITREE_OR_WAIT_CAMERAS"] = ",".join(camera_names)
    env["UNITREE_OR_CAMERA_READY_TIMEOUT"] = str(timeout_s)
    env["UNITREE_DDS_DOMAIN_ID"] = str(DDS_CHANNEL)
    env["UNITREE_DDS_INTERFACE"] = DDS_INTERFACE
    env["UNITREE_CAMERA_TRANSPORT"] = OR_CAMERA_TRANSPORT
    result = subprocess.run(
        [detect_sim_python(), "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s + 15.0,
    )
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    return result.returncode == 0


def wait_for_camera_images(camera_names: str, timeout_s: float = CAMERA_READY_TIMEOUT) -> bool:
    requested = [name.strip() for name in camera_names.split(",") if name.strip()]
    if not requested:
        return True
    print(f"[or-vlm] waiting for DDS cameras: {', '.join(requested)}", flush=True)
    if _wait_for_dds_camera_images(requested, timeout_s):
        return True
    print(f"[or-vlm] DDS camera wait timed out; missing: {', '.join(requested[:8])}")
    return False


def start_and_execute_structured_agent(default_scene: str = "halo") -> int:
    scene = select_or_scene(default_scene)
    if scene is None:
        return 0

    code = start_simulator(scene)
    if code != 0:
        return code

    if not wait_for_camera_images(f"head,scene_00", timeout_s=CAMERA_READY_TIMEOUT):
        return 2
    instruction = input("agent instruction> ").strip()
    if not instruction:
        print("[or-vlm] missing instruction")
        return 2
    return execute_structured_state_agent(instruction)


def execute_wasd_teleop() -> int:
    pid = read_pid()
    if pid is None or not is_pid_alive(pid):
        print("[or-vlm] simulator is not running; start it first if you want the robot to respond")

    command = [
        detect_sim_python(),
        str(WASD_TELEOP),
        "--channel-id",
        str(DDS_CHANNEL),
        "--dds-interface",
        DDS_INTERFACE,
    ]
    print("[or-vlm] WASD teleop mode", flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def menu() -> int:
    print("[or-vlm] OR VLM navigation menu")
    try:
        while True:
            current_scene = read_metadata().get("scene", "halo")
            if current_scene not in OR_SCENES:
                current_scene = "halo"
            print()
            print("1) Start structured-state agent navigation")
            print("2) Start simulator only")
            print("3) Execute structured-state instruction on running simulator")
            print("4) WASD walk")
            print("5) Stop simulator")
            print("q) Quit")
            choice = input("select> ").strip().lower()

            if choice in {"q", "quit", "exit"}:
                print("[or-vlm] quitting; stopping managed simulator")
                return stop_simulator()
            if choice == "1":
                start_and_execute_structured_agent(current_scene)
            elif choice == "2":
                scene = select_or_scene(current_scene)
                if scene is not None:
                    start_simulator(scene)
            elif choice == "3":
                execute_structured_state_agent(input("agent instruction> "))
            elif choice == "4":
                execute_wasd_teleop()
            elif choice == "5":
                stop_simulator()
            else:
                print("[or-vlm] unknown selection")
    except (EOFError, KeyboardInterrupt):
        print()
        print("[or-vlm] exiting; stopping managed simulator")
        return stop_simulator()


def main() -> int:
    return menu()


if __name__ == "__main__":
    raise SystemExit(main())
