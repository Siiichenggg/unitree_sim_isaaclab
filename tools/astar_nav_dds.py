#!/usr/bin/env python3

"""A* navigation command publisher for OR wholebody tasks.

The script builds a 2D occupancy grid, plans an A* path, and publishes
Unitree wholebody run commands on ``rt/run_command/cmd``:

    [forward_velocity, lateral_velocity, yaw_velocity, body_height]

It can subscribe to ``rt/sim_state`` for closed-loop robot pose feedback. Use
``--plan-only`` to validate a path without DDS.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_OBJS = {
    "halo": PROJECT_ROOT / "assets" / "objects" / "OR" / "Model" / "halo_room_baked" / "halo_room_baked.obj",
    "pulm": PROJECT_ROOT / "assets" / "objects" / "OR" / "Model" / "pulm_room_baked" / "pulm_room_baked.obj",
}
DEFAULT_STARTS = {
    "halo": (1.35, -1.45),
    "pulm": (2.10, -1.20),
}


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class OccupancyGrid:
    occupied: np.ndarray
    bounds: Bounds
    resolution: float

    @property
    def width(self) -> int:
        return int(self.occupied.shape[1])

    @property
    def height(self) -> int:
        return int(self.occupied.shape[0])
        return int(self.occupied.shape[0])

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(round((x - self.bounds.xmin) / self.resolution))
        row = int(round((y - self.bounds.ymin) / self.resolution))
        return row, col
    
    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.bounds.xmin + col * self.resolution
        y = self.bounds.ymin + row * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and not bool(self.occupied[row, col])


class SimPoseSubscriber:
    """Subscribe to ``rt/sim_state`` and expose the latest robot XY-yaw pose."""

    def __init__(self, channel_id: int = 0, dds_interface: str = "lo"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self._lock = threading.Lock()
        self._pose: tuple[float, float, float] | None = None
        self._last_time = 0.0

        if dds_interface:
            ChannelFactoryInitialize(channel_id, dds_interface)
        else:
            ChannelFactoryInitialize(channel_id)
        self._subscriber = ChannelSubscriber("rt/sim_state", String_)
        self._subscriber.Init(lambda msg: self._callback(msg), 1)

    def _callback(self, msg) -> None:
        pose = extract_robot_pose_from_sim_state(msg.data)
        if pose is None:
            return
        with self._lock:
            self._pose = pose
            self._last_time = time.monotonic()

    def latest(self) -> tuple[float, float, float] | None:
        with self._lock:
            return self._pose

    def age(self) -> float:
        with self._lock:
            if self._last_time == 0.0:
                return math.inf
            return time.monotonic() - self._last_time


class RunCommandPublisher:
    """Publish wholebody run commands to ``rt/run_command/cmd``."""

    def __init__(self, channel_id: int = 0, dds_interface: str = "lo"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self._msg_cls = String_
        if dds_interface:
            ChannelFactoryInitialize(channel_id, dds_interface)
        else:
            ChannelFactoryInitialize(channel_id)
        self._publisher = ChannelPublisher("rt/run_command/cmd", String_)
        self._publisher.Init()

    def publish(self, vx: float, vy: float, yaw_rate: float, height: float) -> None:
        command = [float(vx), float(vy), float(yaw_rate), float(height)]
        self._publisher.Write(self._msg_cls(data=str(command)))

    def stop(self, height: float = 0.8) -> None:
        self.publish(0.0, 0.0, 0.0, height)


def parse_pair(values: list[str] | tuple[str, ...], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"{name} expects two values: X Y")
    return float(values[0]), float(values[1])


def parse_bounds(values: list[str] | None) -> Bounds | None:
    if values is None:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--bounds expects four values: XMIN YMIN XMAX YMAX")
    xmin, ymin, xmax, ymax = (float(v) for v in values)
    if not xmin < xmax or not ymin < ymax:
        raise argparse.ArgumentTypeError("--bounds must satisfy XMIN < XMAX and YMIN < YMAX")
    return Bounds(xmin, ymin, xmax, ymax)


def parse_rect(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rectangles must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if not xmin < xmax or not ymin < ymax:
        raise argparse.ArgumentTypeError("rectangle must satisfy xmin < xmax and ymin < ymax")
    return xmin, ymin, xmax, ymax


def extract_robot_pose_from_sim_state(raw: str) -> tuple[float, float, float] | None:
    """Extract robot root pose from sim_main's published sim_state JSON."""
    try:
        outer = json.loads(raw)
        state = outer.get("init_state", outer)
        if isinstance(state, str):
            state = json.loads(state)
        root_pose = state["articulation"]["robot"]["root_pose"][0]
        x, y = float(root_pose[0]), float(root_pose[1])
        qw, qx, qy, qz = (float(v) for v in root_pose[3:7])
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return x, y, yaw
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_obj_occupancy(
    obj_path: Path,
    resolution: float,
    bounds: Bounds | None,
    obstacle_min_z: float,
    obstacle_max_z: float,
    projection: str,
) -> OccupancyGrid:
    """Project low-height OBJ geometry into a coarse 2D occupancy grid."""
    if not obj_path.is_file():
        raise FileNotFoundError(obj_path)

    vertices: list[tuple[float, float, float]] = []
    raw_faces: list[list[int]] = []
    xmin = ymin = math.inf
    xmax = ymax = -math.inf

    with obj_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                _, xs, ys, zs, *_ = line.split()
                x, y, z = float(xs), float(ys), float(zs)
                vertices.append((x, y, z))
                xmin, ymin = min(xmin, x), min(ymin, y)
                xmax, ymax = max(xmax, x), max(ymax, y)
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    vertex_token = token.split("/", 1)[0]
                    if not vertex_token:
                        continue
                    index = int(vertex_token)
                    if index < 0:
                        index = len(vertices) + index + 1
                    indices.append(index - 1)
                if len(indices) >= 3:
                    raw_faces.append(indices)

    if not vertices or not raw_faces:
        raise RuntimeError(f"No vertices/faces found in {obj_path}")

    grid_bounds = bounds or Bounds(xmin, ymin, xmax, ymax)
    width = int(math.ceil((grid_bounds.xmax - grid_bounds.xmin) / resolution)) + 1
    height = int(math.ceil((grid_bounds.ymax - grid_bounds.ymin) / resolution)) + 1
    occupied = np.zeros((height, width), dtype=bool)

    def mark_cell(x: float, y: float) -> None:
        col = int(round((x - grid_bounds.xmin) / resolution))
        row = int(round((y - grid_bounds.ymin) / resolution))
        if 0 <= row < height and 0 <= col < width:
            occupied[row, col] = True

    def mark_bbox(face_vertices: list[tuple[float, float, float]]) -> None:
        z_values = [v[2] for v in face_vertices]
        if max(z_values) < obstacle_min_z or min(z_values) > obstacle_max_z:
            return
        xs = [v[0] for v in face_vertices]
        ys = [v[1] for v in face_vertices]
        min_col = int(math.floor((min(xs) - grid_bounds.xmin) / resolution))
        max_col = int(math.ceil((max(xs) - grid_bounds.xmin) / resolution))
        min_row = int(math.floor((min(ys) - grid_bounds.ymin) / resolution))
        max_row = int(math.ceil((max(ys) - grid_bounds.ymin) / resolution))
        min_col, max_col = max(min_col, 0), min(max_col, width - 1)
        min_row, max_row = max(min_row, 0), min(max_row, height - 1)
        if min_col <= max_col and min_row <= max_row:
            occupied[min_row : max_row + 1, min_col : max_col + 1] = True

    if projection == "vertices":
        for x, y, z in vertices:
            if obstacle_min_z <= z <= obstacle_max_z:
                mark_cell(x, y)
    elif projection == "faces":
        for face in raw_faces:
            try:
                mark_bbox([vertices[index] for index in face])
            except IndexError:
                continue
    else:
        raise ValueError(f"Unsupported projection mode: {projection}")

    return OccupancyGrid(occupied=occupied, bounds=grid_bounds, resolution=resolution)


def make_empty_grid(bounds: Bounds, resolution: float) -> OccupancyGrid:
    width = int(math.ceil((bounds.xmax - bounds.xmin) / resolution)) + 1
    height = int(math.ceil((bounds.ymax - bounds.ymin) / resolution)) + 1
    return OccupancyGrid(occupied=np.zeros((height, width), dtype=bool), bounds=bounds, resolution=resolution)


def mark_rectangles(grid: OccupancyGrid, rectangles: Iterable[tuple[float, float, float, float]]) -> None:
    for xmin, ymin, xmax, ymax in rectangles:
        min_row, min_col = grid.world_to_cell(xmin, ymin)
        max_row, max_col = grid.world_to_cell(xmax, ymax)
        min_row, max_row = sorted((min_row, max_row))
        min_col, max_col = sorted((min_col, max_col))
        min_row, max_row = max(min_row, 0), min(max_row, grid.height - 1)
        min_col, max_col = max(min_col, 0), min(max_col, grid.width - 1)
        if min_row <= max_row and min_col <= max_col:
            grid.occupied[min_row : max_row + 1, min_col : max_col + 1] = True


def inflate_obstacles(grid: OccupancyGrid, radius: float) -> None:
    cells = int(math.ceil(radius / grid.resolution))
    if cells <= 0:
        return

    base = grid.occupied.copy()
    inflated = base.copy()
    offsets = [
        (dr, dc)
        for dr in range(-cells, cells + 1)
        for dc in range(-cells, cells + 1)
        if math.hypot(dr, dc) <= cells
    ]
    occupied_rows, occupied_cols = np.nonzero(base)
    for dr, dc in offsets:
        rows = occupied_rows + dr
        cols = occupied_cols + dc
        valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        inflated[rows[valid], cols[valid]] = True
    grid.occupied = inflated


def clear_world_radius(grid: OccupancyGrid, center_xy: tuple[float, float], radius: float) -> None:
    if radius <= 0.0:
        return
    center_row, center_col = grid.world_to_cell(*center_xy)
    cells = int(math.ceil(radius / grid.resolution))
    for dr in range(-cells, cells + 1):
        for dc in range(-cells, cells + 1):
            if math.hypot(dr, dc) * grid.resolution > radius:
                continue
            row, col = center_row + dr, center_col + dc
            if grid.in_bounds(row, col):
                grid.occupied[row, col] = False


def nearest_free_cell(grid: OccupancyGrid, row: int, col: int) -> tuple[int, int]:
    if grid.is_free(row, col):
        return row, col
    max_radius = max(grid.width, grid.height)
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in (-radius, radius):
                candidate = row + dr, col + dc
                if grid.is_free(*candidate):
                    return candidate
        for dc in range(-radius + 1, radius):
            for dr in (-radius, radius):
                candidate = row + dr, col + dc
                if grid.is_free(*candidate):
                    return candidate
    raise RuntimeError("No free cell exists in the occupancy grid")


def astar(grid: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        for dr, dc, step_cost in neighbors:
            nxt = current[0] + dr, current[1] + dc
            if not grid.is_free(*nxt) or nxt in closed:
                continue
            tentative = cost + step_cost
            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = current
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nxt), tentative, nxt))

    raise RuntimeError("A* failed to find a path")


def has_line_of_sight(grid: OccupancyGrid, a: tuple[int, int], b: tuple[int, int]) -> bool:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        if not grid.is_free(r0, c0):
            return False
        if (r0, c0) == (r1, c1):
            return True
        err2 = 2 * err
        if err2 > -dr:
            err -= dr
            c0 += sc
        if err2 < dc:
            err += dc
            r0 += sr


def smooth_path(grid: OccupancyGrid, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        next_index = len(path) - 1
        while next_index > anchor + 1 and not has_line_of_sight(grid, path[anchor], path[next_index]):
            next_index -= 1
        smoothed.append(path[next_index])
        anchor = next_index
    return smoothed


def plan_path(
    grid: OccupancyGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    smoothing: bool,
    clear_radius: float,
) -> list[tuple[float, float]]:
    clear_world_radius(grid, start_xy, clear_radius)
    clear_world_radius(grid, goal_xy, clear_radius)
    start = nearest_free_cell(grid, *grid.world_to_cell(*start_xy))
    goal = nearest_free_cell(grid, *grid.world_to_cell(*goal_xy))
    raw_path = astar(grid, start, goal)
    cell_path = smooth_path(grid, raw_path) if smoothing else raw_path
    return [grid.cell_to_world(row, col) for row, col in cell_path]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wait_for_pose(subscriber: SimPoseSubscriber, timeout_s: float) -> tuple[float, float, float] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pose = subscriber.latest()
        if pose is not None:
            return pose
        time.sleep(0.05)
    return None


def follow_path(
    path: list[tuple[float, float]],
    publisher: RunCommandPublisher,
    subscriber: SimPoseSubscriber,
    args: argparse.Namespace,
) -> None:
    goal_index = 1 if len(path) > 1 else 0
    final_goal = path[-1]
    running = True

    def handle_signal(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while running:
            pose = subscriber.latest()
            if pose is None or subscriber.age() > args.feedback_timeout:
                publisher.stop(args.height)
                print("[nav] waiting for fresh sim_state...")
                time.sleep(0.2)
                continue

            x, y, yaw = pose
            final_dist = math.hypot(final_goal[0] - x, final_goal[1] - y)
            if final_dist <= args.goal_tolerance:
                print(f"[nav] reached goal: dist={final_dist:.3f} m")
                break

            target = path[goal_index]
            target_dist = math.hypot(target[0] - x, target[1] - y)
            if target_dist <= args.waypoint_tolerance and goal_index < len(path) - 1:
                goal_index += 1
                target = path[goal_index]
                target_dist = math.hypot(target[0] - x, target[1] - y)
                print(f"[nav] waypoint {goal_index}/{len(path)-1}: {target}")

            desired_yaw = math.atan2(target[1] - y, target[0] - x)
            yaw_error = wrap_angle(desired_yaw - yaw)
            yaw_rate = max(-args.max_yaw_rate, min(args.max_yaw_rate, args.yaw_gain * yaw_error))

            if abs(yaw_error) > args.rotate_in_place_angle and not args.allow_strafe:
                vx, vy = 0.0, 0.0
            else:
                speed = min(args.max_speed, args.speed_gain * target_dist)
                if args.allow_strafe:
                    world_vx = speed * math.cos(desired_yaw)
                    world_vy = speed * math.sin(desired_yaw)
                    vx = math.cos(yaw) * world_vx + math.sin(yaw) * world_vy
                    vy = -math.sin(yaw) * world_vx + math.cos(yaw) * world_vy
                    vx = max(-args.max_speed, min(args.max_speed, vx))
                    vy = max(-args.max_lateral_speed, min(args.max_lateral_speed, vy))
                    if target_dist > args.waypoint_tolerance and vx > 0.0:
                        vx = max(args.min_forward_speed, vx)
                else:
                    vx = max(0.0, speed * math.cos(yaw_error))
                    vy = 0.0

            publisher.publish(vx, vy, yaw_rate, args.height)
            print(
                f"[nav] pose=({x:.2f},{y:.2f},{yaw:.2f}) target=({target[0]:.2f},{target[1]:.2f}) "
                f"cmd=[{vx:.2f},{vy:.2f},{yaw_rate:.2f},{args.height:.2f}] dist={final_dist:.2f}",
                flush=True,
            )
            time.sleep(1.0 / args.command_hz)
    finally:
        publisher.stop(args.height)


def build_grid(args: argparse.Namespace) -> OccupancyGrid:
    bounds = parse_bounds(args.bounds)
    if args.scene == "empty":
        if bounds is None:
            bounds = Bounds(-3.0, -3.0, 3.0, 3.0)
        grid = make_empty_grid(bounds, args.resolution)
    else:
        obj_path = Path(args.obj) if args.obj else SCENE_OBJS[args.scene]
        print(f"[nav] loading occupancy from OBJ: {obj_path}")
        grid = load_obj_occupancy(
            obj_path=obj_path,
            resolution=args.resolution,
            bounds=bounds,
            obstacle_min_z=args.obstacle_min_z,
            obstacle_max_z=args.obstacle_max_z,
            projection=args.projection,
        )

    if args.rect:
        mark_rectangles(grid, [parse_rect(rect) for rect in args.rect])
    inflate_obstacles(grid, args.robot_radius + args.clearance)
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A* navigation command publisher for OR wholebody scenes.")
    parser.add_argument("--scene", choices=["halo", "pulm", "empty"], default="halo", help="Built-in scene map.")
    parser.add_argument("--obj", type=str, default=None, help="Override OBJ file used for occupancy projection.")
    parser.add_argument("--start", nargs=2, metavar=("X", "Y"), default=None, help="Planning start XY. Uses scene default if omitted.")
    parser.add_argument("--goal", nargs=2, metavar=("X", "Y"), required=True, help="Navigation goal XY in world coordinates.")
    parser.add_argument("--bounds", nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), default=None, help="Override planning bounds.")
    parser.add_argument("--rect", action="append", default=[], help="Extra obstacle rectangle: xmin,ymin,xmax,ymax. Repeatable.")
    parser.add_argument("--resolution", type=float, default=0.10, help="A* grid resolution in meters.")
    parser.add_argument("--robot-radius", type=float, default=0.20, help="Robot radius used for obstacle inflation.")
    parser.add_argument("--clearance", type=float, default=0.05, help="Extra obstacle clearance in meters.")
    parser.add_argument(
        "--clear-start-goal-radius",
        type=float,
        default=0.45,
        help="Clear occupied cells around start and goal to avoid mesh noise trapping A* endpoints.",
    )
    parser.add_argument("--obstacle-min-z", type=float, default=0.12, help="Ignore OBJ geometry entirely below this height.")
    parser.add_argument("--obstacle-max-z", type=float, default=1.60, help="Ignore OBJ geometry entirely above this height.")
    parser.add_argument(
        "--projection",
        choices=["vertices", "faces"],
        default="vertices",
        help="OBJ-to-occupancy projection mode. vertices is less conservative; faces is more conservative.",
    )
    parser.add_argument("--no-smooth", action="store_true", help="Disable line-of-sight path smoothing.")
    parser.add_argument("--plan-only", action="store_true", help="Only print the A* path; do not use DDS.")
    parser.add_argument("--channel-id", type=int, default=0, help="DDS ChannelFactoryInitialize id.")
    parser.add_argument("--dds-interface", type=str, default="lo", help="DDS network interface. Empty string uses SDK default.")
    parser.add_argument("--height", type=float, default=0.8, help="Wholebody policy height command.")
    parser.add_argument("--command-hz", type=float, default=20.0, help="DDS command publish rate.")
    parser.add_argument("--max-speed", type=float, default=0.35, help="Max forward speed command.")
    parser.add_argument("--min-forward-speed", type=float, default=0.18, help="Minimum forward speed while strafing toward a waypoint.")
    parser.add_argument("--max-lateral-speed", type=float, default=0.20, help="Max lateral speed command when --allow-strafe is set.")
    parser.add_argument("--max-yaw-rate", type=float, default=0.8, help="Max yaw-rate command.")
    parser.add_argument("--speed-gain", type=float, default=0.8, help="Distance-to-speed proportional gain.")
    parser.add_argument("--yaw-gain", type=float, default=1.5, help="Yaw-error proportional gain.")
    parser.set_defaults(allow_strafe=True)
    parser.add_argument("--allow-strafe", dest="allow_strafe", action="store_true", help="Use lateral velocity and walk while turning.")
    parser.add_argument("--no-strafe", dest="allow_strafe", action="store_false", help="Use mostly turn-then-forward walking.")
    parser.add_argument("--rotate-in-place-angle", type=float, default=0.55, help="Rotate in place when yaw error is above this value and --no-strafe is used.")
    parser.add_argument("--waypoint-tolerance", type=float, default=0.25, help="Distance tolerance for intermediate waypoints.")
    parser.add_argument("--goal-tolerance", type=float, default=0.35, help="Distance tolerance for the final goal.")
    parser.add_argument("--feedback-timeout", type=float, default=1.0, help="Max accepted sim_state age in seconds.")
    parser.add_argument("--pose-timeout", type=float, default=5.0, help="Seconds to wait for initial sim_state pose.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution <= 0.0:
        raise ValueError("--resolution must be positive")
    if args.command_hz <= 0.0:
        raise ValueError("--command-hz must be positive")

    goal_xy = parse_pair(args.goal, "--goal")
    start_xy = parse_pair(args.start, "--start") if args.start is not None else DEFAULT_STARTS.get(args.scene, (0.0, 0.0))

    grid = build_grid(args)

    if not args.plan_only:
        subscriber = SimPoseSubscriber(channel_id=args.channel_id, dds_interface=args.dds_interface)
        pose = wait_for_pose(subscriber, args.pose_timeout)
        if pose is not None:
            start_xy = (pose[0], pose[1])
            print(f"[nav] using current sim_state pose as start: ({start_xy[0]:.3f}, {start_xy[1]:.3f})")
        else:
            print(f"[nav] no sim_state received in {args.pose_timeout:.1f}s; planning from --start/default only")
    else:
        subscriber = None

    path = plan_path(
        grid,
        start_xy=start_xy,
        goal_xy=goal_xy,
        smoothing=not args.no_smooth,
        clear_radius=args.clear_start_goal_radius,
    )
    print(f"[nav] planned {len(path)} waypoints:")
    for index, waypoint in enumerate(path):
        print(f"  {index:02d}: ({waypoint[0]:.3f}, {waypoint[1]:.3f})")

    if args.plan_only:
        return

    if subscriber is None:
        raise RuntimeError("Internal error: subscriber was not created")
    publisher = RunCommandPublisher(channel_id=args.channel_id, dds_interface=args.dds_interface)
    follow_path(path, publisher, subscriber, args)


if __name__ == "__main__":
    main()
