#!/usr/bin/env python3

"""Isaac Lab skills used by the OR structured-state agent."""

from __future__ import annotations

import math
import copy
import difflib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools.or_semantic_state import ORAgentState, normalize_label


def _run_subprocess_realtime(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    tee_stdout: bool = True,
    tee_stderr: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a child process while teeing stdout/stderr live and retaining output."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        start_new_session=True,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def tee(pipe, chunks: list[str], stream) -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                chunks.append(line)
                if stream is not None:
                    print(line, end="", file=stream, flush=True)
        finally:
            pipe.close()

    stdout_stream = sys.stdout if tee_stdout else None
    stderr_stream = sys.stderr if tee_stderr else None
    stdout_thread = threading.Thread(target=tee, args=(process.stdout, stdout_chunks, stdout_stream), daemon=True)
    stderr_thread = threading.Thread(target=tee, args=(process.stderr, stderr_chunks, stderr_stream), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.wait()
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        exc.output = "".join(stdout_chunks)
        exc.stderr = "".join(stderr_chunks)
        raise exc

    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


@dataclass
class ORSkillContext:
    channel_id: int = 0
    dds_interface: str = "lo"
    height: float = 0.8
    max_speed: float = 0.35
    yaw_rate: float = 2.0
    goal_tolerance_m: float = 0.25
    viewpoint_distance_m: float = 0.9
    waypoint_tolerance_m: float = 0.25
    navigation_command_duration_s: float = 0.8
    navigation_command_hz: float = 20.0
    navigation_retry_limit: int = 40
    target_arrival_radius_m: float = 0.85
    near_goal_acceptance_m: float = 0.55
    near_goal_stall_attempts: int = 3
    rotate_in_place_angle_rad: float = 0.65
    min_navigation_progress_m: float = 0.03
    min_viewpoint_progress_m: float = 0.35
    map_resolution_m: float = 0.10
    robot_radius_m: float = 0.25
    safety_margin_m: float = 0.20
    dynamic_obstacle_margin_m: float = 0.25
    clear_start_goal_radius_m: float = 0.45
    obstacle_min_z: float = 0.12
    obstacle_max_z: float = 1.60
    obstacle_queries: str = ""
    max_obstacle_detections: int = 7
    max_obstacle_cameras: int = 4
    safety_horizon_s: float = 0.5
    sim_python: str = ""
    model_path: str = "models/vlm/qwen3-vl-2b-instruct"
    max_env_cameras: int = 6
    perception_timeout_s: float = 120.0
    local_approach_duration_s: float = 30.0
    local_approach_retry_limit: int = 10
    local_stop_area_ratio: float = 0.07
    local_slow_area_ratio: float = 0.16
    local_stop_height_ratio: float = 0.35
    local_max_speed_mps: float = 0.28
    local_min_speed_mps: float = 0.08
    local_max_yaw_rate: float = 0.42
    local_yaw_kp: float = 0.55
    local_forward_center_limit: float = 0.35
    pose_subscriber: object | None = None
    command_publisher: object | None = None
    base_grid: object | None = None
    active_grid: object | None = None

    def ensure_runtime(self) -> None:
        if self.pose_subscriber is None or self.command_publisher is None:
            from tools.astar_nav_dds import RunCommandPublisher, SimPoseSubscriber

            if self.pose_subscriber is None:
                self.pose_subscriber = SimPoseSubscriber(
                    channel_id=self.channel_id,
                    dds_interface=self.dds_interface,
                )
            if self.command_publisher is None:
                self.command_publisher = RunCommandPublisher(
                    channel_id=self.channel_id,
                    dds_interface=self.dds_interface,
                )

    def stop(self) -> None:
        if self.command_publisher is not None:
            self.command_publisher.stop(self.height)

    def detect_sim_python(self) -> str:
        if self.sim_python:
            return self.sim_python
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


def _append_error(state: ORAgentState, message: str) -> ORAgentState:
    errors = list(state.get("errors", []))
    errors.append(message)
    return {"errors": errors}


def _extract_target_hint(instruction: str) -> str:
    text = normalize_label(instruction)
    for prefix in ("go to ", "walk to ", "find ", "navigate to ", "move to "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _label_tokens(text: str) -> list[str]:
    stop_words = {"a", "an", "the", "to", "near", "at"}
    return [token for token in normalize_label(text).replace("_", " ").split() if token not in stop_words]


def _label_match_score(hint: str, label: str) -> float:
    hint_tokens = _label_tokens(hint)
    label_tokens = _label_tokens(label)
    if not hint_tokens or not label_tokens:
        return 0.0
    hint_norm = " ".join(hint_tokens)
    label_norm = " ".join(label_tokens)
    if hint_norm == label_norm:
        return 1.0
    if hint_norm in label_norm or label_norm in hint_norm:
        return 0.92
    token_overlap = len(set(hint_tokens) & set(label_tokens)) / max(len(set(hint_tokens) | set(label_tokens)), 1)
    fuzzy = difflib.SequenceMatcher(None, hint_norm, label_norm).ratio()
    return max(token_overlap, fuzzy)


def _pose_rank(obj: dict) -> int:
    if obj.get("world_pose_source") == "multi_view_triangulation" and obj.get("world_xyz") is not None:
        return 2
    if obj.get("world_xy") is not None:
        return 1
    return 0


def _pose_fields(obj: dict) -> dict:
    return {
        key: obj[key]
        for key in ("world_xy", "world_xyz", "world_pose_source", "triangulation_error_m")
        if key in obj
    }


def _restore_pose_fields(obj: dict, fields: dict) -> None:
    for key in ("world_xy", "world_xyz", "world_pose_source", "triangulation_error_m"):
        if key in fields:
            obj[key] = fields[key]
        elif key in obj:
            obj.pop(key, None)


OBJECT_MERGE_DISTANCE_M = 0.65
OBJECT_BBOX_IOU_GATE = 0.35


def _world_xy_tuple(obj: dict) -> tuple[float, float] | None:
    world_xy = obj.get("world_xy")
    if world_xy is None:
        return None
    try:
        return (float(world_xy[0]), float(world_xy[1]))
    except (TypeError, ValueError, IndexError):
        return None


def _bbox_tuple(box: object) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    try:
        x1, y1, x2, y2 = box
        return (float(x1), float(y1), float(x2), float(y2))
    except (TypeError, ValueError):
        return None


def _bbox_iou(a: object, b: object) -> float:
    box_a = _bbox_tuple(a)
    box_b = _bbox_tuple(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _shared_view_bbox_iou(a: dict, b: dict) -> float:
    a_boxes = a.get("bbox_by_view") or {}
    b_boxes = b.get("bbox_by_view") or {}
    if not isinstance(a_boxes, dict) or not isinstance(b_boxes, dict):
        return 0.0
    shared_views = set(a_boxes) & set(b_boxes)
    if not shared_views:
        return 0.0
    return max(_bbox_iou(a_boxes[view], b_boxes[view]) for view in shared_views)


def _object_merge_score(existing: dict, incoming: dict, label: str) -> tuple[int, float] | None:
    if normalize_label(str(existing.get("label", ""))) != label:
        return None

    existing_id = str(existing.get("object_id") or "")
    incoming_id = str(incoming.get("object_id") or "")
    if existing_id and incoming_id and existing_id == incoming_id:
        return (0, 0.0)

    existing_xy = _world_xy_tuple(existing)
    incoming_xy = _world_xy_tuple(incoming)
    if existing_xy is not None and incoming_xy is not None:
        distance = math.hypot(existing_xy[0] - incoming_xy[0], existing_xy[1] - incoming_xy[1])
        if distance <= OBJECT_MERGE_DISTANCE_M:
            return (1, distance)
        return None

    bbox_iou = _shared_view_bbox_iou(existing, incoming)
    if bbox_iou >= OBJECT_BBOX_IOU_GATE:
        return (2, 1.0 - bbox_iou)

    return None


def _merge_objects(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = list(existing)
    for item in incoming:
        label = normalize_label(str(item.get("label", "")))
        world_xy = item.get("world_xy")
        matched = None
        matched_score: tuple[int, float] | None = None
        for obj in merged:
            score = _object_merge_score(obj, item, label)
            if score is None:
                continue
            if matched_score is None or score < matched_score:
                matched = obj
                matched_score = score
        if matched is None:
            merged.append(item)
            continue
        previous_pose_rank = _pose_rank(matched)
        previous_pose_fields = _pose_fields(matched)
        incoming_pose_rank = _pose_rank(item)
        incoming_pose_fields = _pose_fields(item)
        previous_world_xy = matched.get("world_xy")
        previous_seen_by = set(matched.get("seen_by", []))
        previous_bbox_by_view = dict(matched.get("bbox_by_view", {}))
        previous_head_visible = bool(matched.get("head_visible"))
        if float(item.get("confidence", 0.0)) >= float(matched.get("confidence", 0.0)):
            matched.update(item)
            if matched.get("world_xy") is None and previous_world_xy is not None:
                matched["world_xy"] = previous_world_xy
        else:
            if matched.get("world_xy") is None and world_xy is not None:
                matched["world_xy"] = world_xy
        if previous_pose_rank > incoming_pose_rank:
            _restore_pose_fields(matched, previous_pose_fields)
        elif incoming_pose_rank > 0:
            _restore_pose_fields(matched, incoming_pose_fields)
        seen_by = previous_seen_by | set(matched.get("seen_by", [])) | set(item.get("seen_by", []))
        matched["seen_by"] = sorted(seen_by)
        bbox_by_view = previous_bbox_by_view
        bbox_by_view.update(matched.get("bbox_by_view", {}))
        bbox_by_view.update(item.get("bbox_by_view", {}))
        matched["bbox_by_view"] = bbox_by_view
        matched["head_visible"] = previous_head_visible or bool(matched.get("head_visible")) or bool(item.get("head_visible"))
    return merged


def _merge_obstacles(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = list(existing)
    for item in incoming:
        if not item.get("world_xy"):
            continue
        label = normalize_label(str(item.get("query") or item.get("label", "")))
        xy = item.get("world_xy")
        matched = None
        for obj in merged:
            obj_label = normalize_label(str(obj.get("query") or obj.get("label", "")))
            obj_xy = obj.get("world_xy")
            if obj_label != label or obj_xy is None:
                continue
            if math.hypot(float(obj_xy[0]) - float(xy[0]), float(obj_xy[1]) - float(xy[1])) <= 0.55:
                matched = obj
                break
        if matched is None:
            merged.append(item)
            continue
        if float(item.get("confidence", 0.0)) >= float(matched.get("confidence", 0.0)):
            matched.update(item)
        else:
            seen_by = set(matched.get("seen_by", [])) | set(item.get("seen_by", []))
            matched["seen_by"] = sorted(seen_by)
    return merged


def _is_target_like(obj: dict, target: dict | None, target_label: str) -> bool:
    if target is not None and obj.get("object_id") == target.get("object_id"):
        return True
    label = normalize_label(str(obj.get("query") or obj.get("label", "")))
    target_label = normalize_label(target_label)
    return bool(target_label and (target_label in label or label in target_label))


def _obstacle_rectangles(state: ORAgentState, context: ORSkillContext) -> list[tuple[float, float, float, float]]:
    rectangles: list[tuple[float, float, float, float]] = []
    target = state.get("target_object")
    target_label = str(state.get("target_label") or "")
    for obj in state.get("obstacle_objects", []):
        if not obj.get("is_obstacle", True):
            continue
        if _is_target_like(obj, target, target_label):
            continue
        xy = obj.get("world_xy")
        if xy is None:
            continue
        radius_raw = obj.get("footprint_radius_m")
        if radius_raw is None:
            continue
        try:
            radius = float(radius_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(radius) or radius <= 0.0:
            continue
        radius += context.robot_radius_m + context.safety_margin_m
        if obj.get("dynamic"):
            radius += context.dynamic_obstacle_margin_m
        x, y = float(xy[0]), float(xy[1])
        rectangles.append((x - radius, y - radius, x + radius, y + radius))
    return rectangles


def _base_planning_grid(context: ORSkillContext, scene: str):
    from tools.astar_nav_dds import SCENE_OBJS, inflate_obstacles, load_obj_occupancy

    if context.base_grid is not None:
        return context.base_grid
    obj_path = SCENE_OBJS.get(scene)
    if obj_path is None:
        raise RuntimeError(f"no static map OBJ registered for scene={scene!r}")
    grid = load_obj_occupancy(
        obj_path=obj_path,
        resolution=context.map_resolution_m,
        bounds=None,
        obstacle_min_z=context.obstacle_min_z,
        obstacle_max_z=context.obstacle_max_z,
        projection="vertices",
    )
    inflate_obstacles(grid, context.robot_radius_m + context.safety_margin_m)
    context.base_grid = grid
    return grid


def _planning_grid(state: ORAgentState, context: ORSkillContext):
    from tools.astar_nav_dds import mark_rectangles

    grid = copy.deepcopy(_base_planning_grid(context, state.get("scene", "halo")))
    rectangles = _obstacle_rectangles(state, context)
    if rectangles:
        mark_rectangles(grid, rectangles)
    return grid


def _grid_summary(grid, obstacle_count: int, context: ORSkillContext) -> dict:
    occupied_cells = int(grid.occupied.sum())
    total_cells = int(grid.occupied.size)
    return {
        "resolution_m": grid.resolution,
        "bounds": {
            "xmin": grid.bounds.xmin,
            "ymin": grid.bounds.ymin,
            "xmax": grid.bounds.xmax,
            "ymax": grid.bounds.ymax,
        },
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
        "occupied_fraction": occupied_cells / max(1, total_cells),
        "dynamic_obstacles": obstacle_count,
        "inflation_m": context.robot_radius_m + context.safety_margin_m,
    }


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _candidate_angles(preferred_angle: float) -> list[float]:
    offsets = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]
    return [preferred_angle + math.radians(offset) for offset in offsets]


def _is_rejected_viewpoint(
    state: ORAgentState,
    candidate: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    radius_m: float = 0.55,
) -> bool:
    for item in state.get("rejected_viewpoints", []):
        xy = item.get("viewpoint_xy")
        rejected_target = item.get("target_xy")
        if xy is None:
            continue
        if rejected_target is not None:
            if math.hypot(float(rejected_target[0]) - target_xy[0], float(rejected_target[1]) - target_xy[1]) > 0.75:
                continue
        if math.hypot(float(xy[0]) - candidate[0], float(xy[1]) - candidate[1]) <= radius_m:
            return True
    return False


def _line_of_sight(grid, a_xy: tuple[float, float], b_xy: tuple[float, float]) -> bool:
    from tools.astar_nav_dds import clear_world_radius, has_line_of_sight, nearest_free_cell

    los_grid = copy.deepcopy(grid)
    clear_world_radius(los_grid, a_xy, 0.20)
    clear_world_radius(los_grid, b_xy, 0.35)
    a_cell = nearest_free_cell(los_grid, *los_grid.world_to_cell(*a_xy))
    b_cell = nearest_free_cell(los_grid, *los_grid.world_to_cell(*b_xy))
    return has_line_of_sight(los_grid, a_cell, b_cell)


def _is_cell_safe(grid, xy: tuple[float, float]) -> bool:
    row, col = grid.world_to_cell(*xy)
    return grid.is_free(row, col)


def _plan_safe_viewpoint(
    state: ORAgentState,
    context: ORSkillContext,
    target_xy: tuple[float, float],
    robot_pose: tuple[float, float, float],
) -> tuple[tuple[float, float], list[tuple[float, float]], object, dict]:
    from tools.astar_nav_dds import plan_path

    rx, ry, robot_yaw = robot_pose
    tx, ty = target_xy
    preferred = math.atan2(ry - ty, rx - tx)
    radii = [
        context.viewpoint_distance_m,
        context.viewpoint_distance_m + 0.35,
        context.viewpoint_distance_m + 0.70,
    ]
    best: tuple[float, tuple[float, float], list[tuple[float, float]], object, dict] | None = None
    best_rank: tuple[int, float] | None = None
    for radius in radii:
        for angle in _candidate_angles(preferred):
            candidate = (tx + radius * math.cos(angle), ty + radius * math.sin(angle))
            if _is_rejected_viewpoint(state, candidate, target_xy):
                continue
            grid = _planning_grid(state, context)
            if not _is_cell_safe(grid, candidate):
                continue
            try:
                path = plan_path(
                    grid,
                    start_xy=(rx, ry),
                    goal_xy=candidate,
                    smoothing=True,
                    clear_radius=context.clear_start_goal_radius_m,
                )
            except Exception:
                continue
            visible = _line_of_sight(grid, candidate, target_xy)
            path_cost = _path_length(path)
            distance_cost = math.hypot(candidate[0] - rx, candidate[1] - ry)
            if not visible and distance_cost < context.min_viewpoint_progress_m:
                continue
            head_yaw = math.atan2(ty - candidate[1], tx - candidate[0])
            head_camera_angle_cost = abs(_wrap_angle(head_yaw - robot_yaw)) * 0.15
            collision_cost = 0.0 if visible else 2.5
            score = path_cost + 0.25 * distance_cost + head_camera_angle_cost + collision_cost
            rank = (0 if visible else 1, score)
            details = {
                "viewpoint_xy": candidate,
                "target_xy": target_xy,
                "visibility_score": 1.0 if visible else 0.0,
                "collision_cost": collision_cost,
                "distance_cost": distance_cost,
                "head_camera_angle_cost": head_camera_angle_cost,
                "path_length_m": path_cost,
                "score": score,
                "rank": rank,
            }
            if best is None or best_rank is None or rank < best_rank:
                best = (score, candidate, path, grid, details)
                best_rank = rank
    if best is None:
        raise RuntimeError("no reachable safe visible viewpoint")
    _, goal, path, grid, details = best
    return goal, path, grid, details


def _near_occupied(grid, xy: tuple[float, float], radius_m: float) -> bool:
    row, col = grid.world_to_cell(*xy)
    cells = max(1, int(math.ceil(radius_m / grid.resolution)))
    for dr in range(-cells, cells + 1):
        for dc in range(-cells, cells + 1):
            if math.hypot(dr, dc) * grid.resolution > radius_m:
                continue
            if grid.in_bounds(row + dr, col + dc) and bool(grid.occupied[row + dr, col + dc]):
                return True
    return False


def _safety_filter(
    cmd: tuple[float, float, float],
    pose: tuple[float, float, float],
    context: ORSkillContext,
) -> tuple[tuple[float, float, float], str]:
    grid = context.active_grid
    if grid is None:
        return cmd, "safe_no_grid"
    vx, vy, yaw_rate = cmd
    x, y, yaw = pose
    steps = 5
    dt = context.safety_horizon_s / steps
    near_obstacle = False
    for step in range(1, steps + 1):
        future_yaw = yaw + yaw_rate * dt * step
        world_vx = math.cos(future_yaw) * vx - math.sin(future_yaw) * vy
        world_vy = math.sin(future_yaw) * vx + math.cos(future_yaw) * vy
        px = x + world_vx * dt * step
        py = y + world_vy * dt * step
        row, col = grid.world_to_cell(px, py)
        if not grid.is_free(row, col):
            return (0.0, 0.0, 0.0), "safety_stop_predicted_collision"
        near_obstacle = near_obstacle or _near_occupied(grid, (px, py), context.map_resolution_m * 1.5)
    if near_obstacle:
        return (vx * 0.45, vy * 0.45, yaw_rate * 0.7), "safety_slow_near_obstacle"
    return cmd, "safe"


def _waypoint_command(
    context: ORSkillContext,
    pose: tuple[float, float, float],
    waypoint: tuple[float, float],
) -> tuple[tuple[float, float, float], float, float]:
    rx, ry, yaw = pose
    wx, wy = waypoint
    waypoint_distance = math.hypot(wx - rx, wy - ry)
    target_yaw = math.atan2(wy - ry, wx - rx)
    yaw_error = _wrap_angle(target_yaw - yaw)
    yaw_rate = max(-context.yaw_rate, min(context.yaw_rate, 1.15 * yaw_error))
    if abs(yaw_error) > context.rotate_in_place_angle_rad:
        vx = 0.0
    else:
        speed = min(context.max_speed, waypoint_distance * 0.45)
        if waypoint_distance > context.near_goal_acceptance_m:
            speed = max(0.08, speed)
        else:
            speed = min(0.12, speed)
        vx = min(context.max_speed, max(0.0, speed * max(0.0, math.cos(yaw_error))))
    return (vx, 0.0, yaw_rate), yaw_error, waypoint_distance


def _publish_navigation_command(
    context: ORSkillContext,
    pose: tuple[float, float, float],
    waypoint: tuple[float, float],
) -> tuple[str, tuple[float, float, float] | None, dict]:
    """Publish a short closed-loop command window toward one waypoint."""
    duration = max(0.0, context.navigation_command_duration_s)
    period = 1.0 / max(1.0, context.navigation_command_hz)
    deadline = time.monotonic() + duration
    safety_status = "safe"
    latest_pose: tuple[float, float, float] | None = pose
    debug = {
        "target_xy": waypoint,
        "yaw_error": None,
        "cmd": (0.0, 0.0, 0.0),
        "safety_status": safety_status,
        "waypoint_distance": None,
    }
    while time.monotonic() < deadline:
        live_pose = pose
        if context.pose_subscriber is not None:
            latest = context.pose_subscriber.latest()
            if latest is not None:
                live_pose = latest
                latest_pose = latest
        cmd, yaw_error, waypoint_distance = _waypoint_command(context, live_pose, waypoint)
        safe_cmd, safety_status = _safety_filter(cmd, live_pose, context)
        debug = {
            "target_xy": waypoint,
            "yaw_error": yaw_error,
            "cmd": safe_cmd,
            "raw_cmd": cmd,
            "safety_status": safety_status,
            "waypoint_distance": waypoint_distance,
        }
        if safety_status == "safety_stop_predicted_collision":
            context.stop()
            return safety_status, latest_pose, debug
        if context.command_publisher is not None:
            context.command_publisher.publish(*safe_cmd, context.height)
        time.sleep(period)
    if context.pose_subscriber is not None:
        latest = context.pose_subscriber.latest()
        if latest is not None:
            latest_pose = latest
    return safety_status, latest_pose, debug


def _run_perception_skill(state: ORAgentState, context: ORSkillContext, *, mode: str) -> dict:
    if mode == "global":
        env_cameras = ",".join(f"scene_{index:02d}" for index in range(11))
        head_cameras = ""
    elif mode == "head":
        env_cameras = ""
        head_cameras = "head"
    else:
        env_cameras = ",".join(f"scene_{index:02d}" for index in range(11))
        head_cameras = "head"
    project_root = Path(__file__).resolve().parents[1]
    command = [
        context.detect_sim_python(),
        str(project_root / "tools" / "or_scene_perception.py"),
        state.get("instruction", ""),
        "--scene",
        state.get("scene", "halo"),
        "--room-usd",
        state.get("room_usd", ""),
        "--model-path",
        str(project_root / context.model_path),
        "--env-cameras",
        env_cameras,
        "--head-cameras",
        head_cameras,
        "--max-env-cameras",
        str(context.max_env_cameras),
        "--obstacle-queries",
        context.obstacle_queries if mode == "global" else "",
        "--max-obstacle-detections",
        str(context.max_obstacle_detections if mode == "global" else 0),
        "--max-obstacle-cameras",
        str(context.max_obstacle_cameras if mode == "global" else 0),
    ]
    result = _run_subprocess_realtime(
        command,
        cwd=project_root,
        timeout=context.perception_timeout_s,
        tee_stdout=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"perception skill failed code={result.returncode}: {result.stderr[-500:]}")
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("perception skill did not emit JSON")


def observe_global_scene(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    """Update global semantic state from fixed OR cameras.

    Environment cameras are authoritative for global target localization; head
    camera detections are used only to mark local confirmation availability.
    """
    objects = list(state.get("semantic_objects", []))
    obstacles = list(state.get("obstacle_objects", []))
    try:
        perception = _run_perception_skill(state, context, mode="global")
        detected = perception.get("objects", [])
        if detected:
            objects = _merge_objects(objects, detected)
        detected_obstacles = perception.get("obstacle_objects", [])
        if detected_obstacles:
            obstacles = _merge_obstacles(obstacles, detected_obstacles)
    except Exception as exc:
        update = _append_error(state, f"global perception failed: {exc}")
        update.update(
            {
                "semantic_objects": objects,
                "obstacle_objects": obstacles,
                "phase": "robot_localization",
                "last_action": "observe_global_scene_failed",
            }
        )
        return update
    if not objects:
        return {
            "semantic_objects": objects,
            "obstacle_objects": obstacles,
            "occupancy_map": state.get("occupancy_map", {}),
            "phase": "global_perception",
            "last_action": "observe_global_scene_no_target",
        }
    occupancy_map = state.get("occupancy_map", {})
    try:
        grid = _planning_grid({"scene": state.get("scene", "halo"), "obstacle_objects": obstacles}, context)
        occupancy_map = _grid_summary(grid, len(obstacles), context)
    except Exception as exc:
        update = _append_error(state, f"occupancy map build failed: {exc}")
        occupancy_map = dict(occupancy_map)
        occupancy_map["error"] = str(exc)
        return {
            **update,
            "semantic_objects": objects,
            "obstacle_objects": obstacles,
            "occupancy_map": occupancy_map,
            "phase": "robot_localization",
            "last_action": "observe_global_scene_map_failed",
        }
    return {
        "semantic_objects": objects,
        "obstacle_objects": obstacles,
        "occupancy_map": occupancy_map,
        "phase": "robot_localization",
        "last_action": "observe_global_scene",
    }


def localize_robot(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    try:
        context.ensure_runtime()
        pose = context.pose_subscriber.latest() if context.pose_subscriber is not None else None
        deadline = time.monotonic() + 2.0
        while pose is None and time.monotonic() < deadline:
            time.sleep(0.05)
            pose = context.pose_subscriber.latest() if context.pose_subscriber is not None else None
        return {
            "robot_pose": pose,
            "phase": "target_query",
            "last_action": "localize_robot",
        }
    except Exception as exc:
        update = _append_error(state, f"robot localization failed: {exc}")
        update.update({"phase": "target_query", "last_action": "localize_robot_failed"})
        return update


def query_target(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    hint = _extract_target_hint(state.get("instruction", ""))
    hint_norm = normalize_label(hint)
    best = None
    best_rank: tuple[float, float, float, float] | None = None
    for obj in state.get("semantic_objects", []):
        label = str(obj.get("label", ""))
        match_score = _label_match_score(hint_norm, label)
        confidence = float(obj.get("confidence", 0.0))
        seen_count = float(len(obj.get("seen_by", [])))
        has_world_xy = 1.0 if obj.get("world_xy") is not None else 0.0
        rank = (match_score, has_world_xy, min(seen_count, 12.0) / 12.0, confidence)
        if best_rank is None or rank > best_rank:
            best = obj
            best_rank = rank
    if best is None and state.get("semantic_objects"):
        best = state["semantic_objects"][0]
    return {
        "target_label": hint_norm,
        "target_object": best,
        "phase": "viewpoint_planning",
        "last_action": "query_target",
    }


def plan_viewpoint(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    target = state.get("target_object")
    if not target:
        return {"phase": "global_perception", "last_action": "plan_viewpoint_no_target"}
    if target.get("head_visible"):
        return {"head_visible": True, "phase": "local_approach", "last_action": "plan_viewpoint_head_visible"}
    world_xy = target.get("world_xy")
    if world_xy is not None:
        pose = state.get("robot_pose")
        if pose is not None:
            try:
                goal_xy, path, grid, details = _plan_safe_viewpoint(
                    state,
                    context,
                    (float(world_xy[0]), float(world_xy[1])),
                    (float(pose[0]), float(pose[1]), float(pose[2])),
                )
                context.active_grid = grid
                occupancy_map = _grid_summary(grid, len(state.get("obstacle_objects", [])), context)
                occupancy_map["selected_viewpoint"] = details
                return {
                    "nav_goal_xy": goal_xy,
                    "planned_path": path,
                    "active_waypoint_index": 1 if len(path) > 1 else 0,
                    "navigation_attempts": 0,
                    "last_nav_goal_distance": None,
                    "occupancy_map": occupancy_map,
                    "phase": "navigation_execution",
                    "last_action": "plan_safe_visible_viewpoint",
                }
            except Exception as exc:
                update = _append_error(state, f"safe viewpoint planning failed: {exc}")
                update.update(
                    {
                        "nav_goal_xy": None,
                        "planned_path": [],
                        "active_waypoint_index": 0,
                        "navigation_attempts": 0,
                        "last_nav_goal_distance": None,
                        "phase": "global_perception",
                        "last_action": "plan_safe_viewpoint_failed_reobserve",
                    }
                )
                return update
        return {
            "nav_goal_xy": (float(world_xy[0]), float(world_xy[1])),
            "planned_path": [],
            "active_waypoint_index": 0,
            "navigation_attempts": 0,
            "last_nav_goal_distance": None,
            "phase": "navigation_execution",
            "last_action": "plan_visible_viewpoint",
        }
    return {
        "nav_goal_xy": None,
        "phase": "head_camera_search",
        "last_action": "plan_viewpoint_search",
    }


def execute_navigation(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    pose = state.get("robot_pose")
    goal = state.get("nav_goal_xy")
    if pose is None or goal is None:
        return {"phase": "head_camera_search", "last_action": "navigate_missing_pose_or_goal"}
    try:
        context.ensure_runtime()
        rx, ry, yaw = pose
        gx, gy = goal
        path = [(float(x), float(y)) for x, y in state.get("planned_path", [])]
        waypoint_index = int(state.get("active_waypoint_index", 0))
        if not path:
            path = [(rx, ry), (gx, gy)]
            waypoint_index = 1
        waypoint_index = min(max(waypoint_index, 0), len(path) - 1)
        final_goal = path[-1]
        target = state.get("target_object") or {}
        target_xy_raw = target.get("world_xy")
        target_xy = None
        target_distance = None
        if target_xy_raw is not None:
            try:
                target_xy = (float(target_xy_raw[0]), float(target_xy_raw[1]))
                target_distance = math.hypot(target_xy[0] - rx, target_xy[1] - ry)
            except (TypeError, ValueError, IndexError):
                target_xy = None
                target_distance = None
        final_distance = math.hypot(final_goal[0] - rx, final_goal[1] - ry)
        if target_distance is not None and target_distance <= context.target_arrival_radius_m:
            context.stop()
            return {
                "phase": "head_camera_search",
                "robot_pose": pose,
                "active_waypoint_index": waypoint_index,
                "navigation_attempts": 0,
                "last_nav_goal_distance": None,
                "last_action": "navigate_target_area_reached",
            }
        if final_distance <= context.goal_tolerance_m:
            context.stop()
            return {
                "phase": "head_camera_search",
                "active_waypoint_index": waypoint_index,
                "navigation_attempts": 0,
                "last_nav_goal_distance": None,
                "last_action": "navigate_goal_reached",
            }
        attempts = int(state.get("navigation_attempts", 0))
        if attempts >= context.navigation_retry_limit:
            context.stop()
            return {
                "phase": "global_perception",
                "robot_pose": pose,
                "active_waypoint_index": waypoint_index,
                "navigation_attempts": 0,
                "last_nav_goal_distance": None,
                "last_action": "navigate_retry_limit_replan",
            }
        waypoint = path[waypoint_index]
        waypoint_distance = math.hypot(waypoint[0] - rx, waypoint[1] - ry)
        while waypoint_distance <= context.waypoint_tolerance_m and waypoint_index < len(path) - 1:
            waypoint_index += 1
            waypoint = path[waypoint_index]
            waypoint_distance = math.hypot(waypoint[0] - rx, waypoint[1] - ry)
        command, yaw_error, waypoint_distance = _waypoint_command(context, pose, waypoint)
        (_safe_vx, _safe_vy, _safe_yaw_rate), safety_status = _safety_filter(command, pose, context)
        if safety_status == "safety_stop_predicted_collision":
            context.stop()
            return {
                "phase": "global_perception",
                "active_waypoint_index": waypoint_index,
                "last_action": safety_status,
            }
        safety_status, latest_pose, command_debug = _publish_navigation_command(context, pose, waypoint)
        if safety_status == "safety_stop_predicted_collision":
            return {
                "phase": "global_perception",
                "robot_pose": latest_pose or pose,
                "active_waypoint_index": waypoint_index,
                "last_action": safety_status,
            }
        if latest_pose is not None:
            latest_distance = math.hypot(final_goal[0] - latest_pose[0], final_goal[1] - latest_pose[1])
            latest_target_distance = None
            if target_xy is not None:
                latest_target_distance = math.hypot(target_xy[0] - latest_pose[0], target_xy[1] - latest_pose[1])
                if latest_target_distance <= context.target_arrival_radius_m:
                    context.stop()
                    return {
                        "phase": "head_camera_search",
                        "robot_pose": latest_pose,
                        "active_waypoint_index": waypoint_index,
                        "navigation_attempts": 0,
                        "last_nav_goal_distance": None,
                        "last_action": "navigate_target_area_reached_after_command",
                    }
            if latest_distance <= context.goal_tolerance_m:
                context.stop()
                return {
                    "phase": "head_camera_search",
                    "robot_pose": latest_pose,
                    "active_waypoint_index": waypoint_index,
                    "navigation_attempts": 0,
                    "last_nav_goal_distance": None,
                    "last_action": "navigate_goal_reached_after_command",
                }
        else:
            latest_distance = final_distance
        previous_distance = state.get("last_nav_goal_distance")
        made_progress = previous_distance is None or latest_distance < float(previous_distance) - context.min_navigation_progress_m
        next_attempts = 0 if made_progress else attempts + 1
        if latest_distance <= context.near_goal_acceptance_m and next_attempts >= context.near_goal_stall_attempts:
            context.stop()
            return {
                "phase": "head_camera_search",
                "robot_pose": latest_pose or pose,
                "active_waypoint_index": waypoint_index,
                "navigation_attempts": 0,
                "last_nav_goal_distance": None,
                "last_action": "navigate_near_goal_stalled_accept",
            }
        return {
            "phase": "navigation_execution",
            "robot_pose": latest_pose or pose,
            "active_waypoint_index": waypoint_index,
            "navigation_attempts": next_attempts,
            "last_nav_goal_distance": latest_distance,
            "last_nav_command": {
                "target_xy": waypoint,
                "yaw_error": command_debug.get("yaw_error", yaw_error),
                "cmd": command_debug.get("cmd", command),
                "safety_status": safety_status,
                "made_progress": made_progress,
                "goal_distance": latest_distance,
                "target_object_distance": latest_target_distance if latest_pose is not None else target_distance,
                "waypoint_distance": command_debug.get("waypoint_distance", waypoint_distance),
            },
            "last_action": f"navigate_path_waypoint_{waypoint_index}_{safety_status}",
        }
    except Exception as exc:
        update = _append_error(state, f"navigation failed: {exc}")
        update.update({"phase": "failed", "status": "failed", "last_action": "navigate_failed"})
        return update


def search_with_head_camera(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    try:
        context.ensure_runtime()
        perception = _run_perception_skill(state, context, mode="head")
        detected = perception.get("objects", [])
        if detected:
            objects = _merge_objects(list(state.get("semantic_objects", [])), detected)
            return {
                "semantic_objects": objects,
                "head_visible": True,
                "phase": "local_approach",
                "last_action": "confirm_with_head_camera",
            }
        if context.command_publisher is not None:
            context.command_publisher.publish(0.0, 0.0, context.yaw_rate, context.height)
        rejected = list(state.get("rejected_viewpoints", []))
        viewpoint_xy = state.get("nav_goal_xy")
        target = state.get("target_object") or {}
        target_xy = target.get("world_xy")
        if viewpoint_xy is not None:
            rejected.append(
                {
                    "viewpoint_xy": (float(viewpoint_xy[0]), float(viewpoint_xy[1])),
                    "target_xy": (float(target_xy[0]), float(target_xy[1])) if target_xy is not None else None,
                    "reason": "head_camera_not_visible",
                    "timestamp_s": time.monotonic(),
                }
            )
            rejected = rejected[-20:]
        return {
            "rejected_viewpoints": rejected,
            "phase": "global_perception",
            "last_action": "search_with_head_camera_reject_viewpoint",
        }
    except Exception as exc:
        update = _append_error(state, f"head search failed: {exc}")
        update.update({"phase": "failed", "status": "failed", "last_action": "head_search_failed"})
        return update


def local_approach(state: ORAgentState, context: ORSkillContext) -> ORAgentState:
    project_root = Path(__file__).resolve().parents[1]
    target_xy = None
    target = state.get("target_object") or {}
    target_xy_raw = target.get("world_xy")
    if target_xy_raw is not None:
        try:
            target_xy = (float(target_xy_raw[0]), float(target_xy_raw[1]))
        except (TypeError, ValueError, IndexError):
            target_xy = None
    command = [
        context.detect_sim_python(),
        str(project_root / "tools" / "visual_vlm_servo.py"),
        state.get("instruction", ""),
        "--planner-mode",
        "bbox",
        "--camera",
        "head",
        "--search-cameras",
        "head",
        "--servo-cameras",
        "head",
        "--duration",
        str(context.local_approach_duration_s),
        "--stop-area-ratio",
        str(context.local_stop_area_ratio),
        "--slow-area-ratio",
        str(context.local_slow_area_ratio),
        "--stop-height-ratio",
        str(context.local_stop_height_ratio),
        "--max-speed",
        str(context.local_max_speed_mps),
        "--min-speed",
        str(context.local_min_speed_mps),
        "--max-yaw-rate",
        str(context.local_max_yaw_rate),
        "--yaw-kp",
        str(context.local_yaw_kp),
        "--forward-center-limit",
        str(context.local_forward_center_limit),
        "--channel-id",
        str(context.channel_id),
        "--dds-interface",
        context.dds_interface,
        "--model-path",
        str(project_root / context.model_path),
    ]
    if target_xy is not None:
        command.extend(
            [
                "--target-world-xy",
                str(target_xy[0]),
                str(target_xy[1]),
                "--target-arrival-radius",
                str(context.target_arrival_radius_m),
            ]
        )
    try:
        result = _run_subprocess_realtime(
            command,
            cwd=project_root,
            timeout=context.local_approach_duration_s + 45.0,
        )
        latest_pose = None
        try:
            context.ensure_runtime()
            latest_pose = context.pose_subscriber.latest() if context.pose_subscriber is not None else None
            deadline = time.monotonic() + 1.0
            while latest_pose is None and time.monotonic() < deadline:
                time.sleep(0.05)
                latest_pose = context.pose_subscriber.latest() if context.pose_subscriber is not None else None
        except Exception:
            latest_pose = None
        if result.returncode == 0:
            update = {
                "phase": "done",
                "status": "done",
                "local_approach_attempts": 0,
                "last_action": "local_approach",
            }
            if latest_pose is not None:
                update["robot_pose"] = latest_pose
            return update
        saw_target = " score=" in result.stdout or "reached visual stop condition" in result.stdout
        attempts = int(state.get("local_approach_attempts", 0)) + 1
        if result.returncode == 1 and saw_target and attempts < context.local_approach_retry_limit:
            update = {
                "phase": "local_approach",
                "head_visible": True,
                "local_approach_attempts": attempts,
                "last_action": f"local_approach_continue_{attempts}",
            }
            if latest_pose is not None:
                update["robot_pose"] = latest_pose
            return update
        update = {
            "phase": "global_perception",
            "local_approach_attempts": attempts,
            "last_action": f"local_approach_retry_code_{result.returncode}",
        }
        if latest_pose is not None:
            update["robot_pose"] = latest_pose
        return update
    except Exception as exc:
        update = _append_error(state, f"local approach failed: {exc}")
        update.update({"phase": "global_perception", "last_action": "local_approach_failed"})
        return update
