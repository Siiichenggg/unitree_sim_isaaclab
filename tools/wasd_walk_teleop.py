#!/usr/bin/env python3

"""Keyboard teleop for Unitree wholebody run commands."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CommandStreamer:
    def __init__(self, publisher, height: float, hz: float):
        self._publisher = publisher
        self._height = float(height)
        self._period = 1.0 / max(float(hz), 1.0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._command = (0.0, 0.0, 0.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, vx: float, vy: float, yaw_rate: float) -> None:
        with self._lock:
            self._command = (float(vx), float(vy), float(yaw_rate))

    def stop_motion(self) -> None:
        self.update(0.0, 0.0, 0.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                vx, vy, yaw_rate = self._command
            self._publisher.publish(vx, vy, yaw_rate, self._height)
            self._stop_event.wait(self._period)

    def close(self) -> None:
        self.stop_motion()
        time.sleep(min(self._period, 0.05))
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._publisher.stop(self._height)


def read_key(timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def command_for_key(key: str, speed: float, lateral_speed: float, yaw_rate: float) -> tuple[float, float, float] | None:
    match key.lower():
        case "w":
            return speed, 0.0, 0.0
        case "s":
            return -speed, 0.0, 0.0
        case "q":
            return 0.0, 0.0, yaw_rate
        case "e":
            return 0.0, 0.0, -yaw_rate
        case "a":
            return 0.0, lateral_speed, 0.0
        case "d":
            return 0.0, -lateral_speed, 0.0
        case "j":
            return 0.0, 0.0, yaw_rate
        case "l":
            return 0.0, 0.0, -yaw_rate
        case " ":
            return 0.0, 0.0, 0.0
        case _:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WASD keyboard teleop over rt/run_command/cmd.")
    parser.add_argument("--channel-id", type=int, default=0)
    parser.add_argument("--dds-interface", default="lo")
    parser.add_argument("--height", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=0.30)
    parser.add_argument("--lateral-speed", type=float, default=0.22)
    parser.add_argument("--yaw-rate", type=float, default=0.8)
    parser.add_argument("--stream-hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not sys.stdin.isatty():
        print("[wasd] stdin is not a TTY; run from an interactive terminal", file=sys.stderr)
        return 2

    from tools.astar_nav_dds import RunCommandPublisher

    publisher = RunCommandPublisher(channel_id=args.channel_id, dds_interface=args.dds_interface)
    streamer = CommandStreamer(publisher, height=args.height, hz=args.stream_hz)
    streamer.start()

    old_settings = termios.tcgetattr(sys.stdin)
    current = (0.0, 0.0, 0.0)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("[wasd] controls: w/s forward/back, a/d strafe, q/e or j/l turn, space stop, x quit", flush=True)
        print(
            f"[wasd] speed={args.speed:.2f} lateral={args.lateral_speed:.2f} "
            f"yaw={args.yaw_rate:.2f} height={args.height:.2f}",
            flush=True,
        )
        while True:
            key = read_key(0.1)
            if key is None:
                continue
            if key.lower() in {"x", "\x03", "\x04"}:
                print("\n[wasd] quit; stopping", flush=True)
                return 0
            command = command_for_key(key, args.speed, args.lateral_speed, args.yaw_rate)
            if command is None:
                continue
            if command != current:
                current = command
                streamer.update(*command)
                print(
                    f"\r[wasd] key={key.lower()} "
                    f"cmd=[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}, {args.height:.2f}]   ",
                    end="",
                    flush=True,
                )
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        streamer.close()


if __name__ == "__main__":
    raise SystemExit(main())
