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
STATE_DIR = PROJECT_ROOT / ".tmp" / "or_vlm_nav"
LOG_DIR = PROJECT_ROOT / "logs" / "or_vlm_nav"
PID_FILE = STATE_DIR / "sim.pid"
META_FILE = STATE_DIR / "sim.json"
SIM_LOG = LOG_DIR / "sim.log"
VISUAL_SERVO = PROJECT_ROOT / "tools" / "visual_vlm_servo.py"
SIM_MAIN = PROJECT_ROOT / "sim_main.py"
OR_SCENES = {
    "halo": PROJECT_ROOT / "assets" / "objects" / "OR" / "Model" / "halo_room_baked" / "halo_room_baked.usd",
    "pulm": PROJECT_ROOT / "assets" / "objects" / "OR" / "Model" / "pulm_room_baked" / "pulm_room_baked.usd",
}

TASK = "Isaac-Move-Cylinder-G129-Dex3-Wholebody-VLM"
HAND_FLAG = "--enable_dex3_dds"
DDS_CHANNEL = 0
DDS_INTERFACE = "lo"
CAMERA_INCLUDE = (
    "front_camera,front_camera_up,front_camera_down,front_camera_left,front_camera_right,"
    "left_wrist_camera,right_wrist_camera"
)
SEARCH_CAMERAS = "head,head_left,head_right,head_up,head_down"


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
        "--camera_include",
        CAMERA_INCLUDE,
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


def execute_vlm_instruction(instruction: str) -> int:
    instruction = instruction.strip()
    if not instruction:
        print("[or-vlm] missing instruction")
        return 2

    command = [
        detect_sim_python(),
        str(VISUAL_SERVO),
        instruction,
        "--camera",
        "head",
        "--search-cameras",
        SEARCH_CAMERAS,
        "--camera-timeout",
        "5.0",
        "--score-threshold",
        "0.04",
        "--max-new-tokens",
        "96",
        "--duration",
        "45.0",
        "--stream-hz",
        "10.0",
        "--channel-id",
        str(DDS_CHANNEL),
        "--dds-interface",
        DDS_INTERFACE,
        "--debug-image",
        str(LOG_DIR / "visual_grounding_debug.jpg"),
    ]

    print(f"[or-vlm] visual instruction={instruction}", flush=True)
    print("[or-vlm] visual mode=execute", flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def menu() -> int:
    print("[or-vlm] OR VLM navigation menu")
    try:
        while True:
            print()
            print("1) Start simulator")
            print("2) Execute VLM instruction")
            print("q) Quit")
            choice = input("select> ").strip().lower()

            if choice in {"q", "quit", "exit"}:
                print("[or-vlm] quitting; stopping managed simulator")
                return stop_simulator()
            if choice == "1":
                current_scene = read_metadata().get("scene", "halo")
                if current_scene not in OR_SCENES:
                    current_scene = "halo"
                scene = select_or_scene(current_scene)
                if scene is not None:
                    start_simulator(scene)
            elif choice == "2":
                execute_vlm_instruction(input("visual instruction> "))
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
