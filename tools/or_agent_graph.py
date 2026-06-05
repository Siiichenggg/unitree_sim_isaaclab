#!/usr/bin/env python3

"""LangGraph-style OR structured-state navigation agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

from langgraph.graph import END, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.or_semantic_state import ORAgentState, initial_state
from tools.or_skills import (
    ORSkillContext,
    execute_navigation,
    local_approach,
    localize_robot,
    observe_global_scene,
    plan_viewpoint,
    query_target,
    search_with_head_camera,
)


NodeFn = Callable[[ORAgentState, ORSkillContext], ORAgentState]


def _merge_state(state: ORAgentState, update: ORAgentState) -> ORAgentState:
    next_state = dict(state)
    next_state.update(update)
    next_state["step_count"] = int(next_state.get("step_count", 0)) + 1
    if next_state["step_count"] >= int(next_state.get("max_steps", 12)) and next_state.get("phase") not in {"done", "failed"}:
        next_state["phase"] = "failed"
        next_state["status"] = "failed"
        errors = list(next_state.get("errors", []))
        errors.append("agent step budget exhausted")
        next_state["errors"] = errors
    return next_state


def _next_node(state: ORAgentState) -> str | None:
    phase = state.get("phase", "global_perception")
    if phase in {"done", "failed"}:
        return None
    return {
        "global_perception": "global_perception",
        "robot_localization": "robot_localization",
        "target_query": "target_query",
        "viewpoint_planning": "viewpoint_planning",
        "navigation_execution": "navigation_execution",
        "head_camera_search": "head_camera_search",
        "local_approach": "local_approach",
    }.get(phase, "global_perception")


def _graph_nodes(context: ORSkillContext) -> dict[str, Callable[[ORAgentState], ORAgentState]]:
    def run_node(name: str, fn: NodeFn, state: ORAgentState) -> ORAgentState:
        print(f"[or-agent] node={name} phase={state.get('phase')}", flush=True)
        return _merge_state(state, fn(state, context))

    return {
        "global_perception": lambda state: run_node("global_perception", observe_global_scene, state),
        "robot_localization": lambda state: run_node("robot_localization", localize_robot, state),
        "target_query": lambda state: run_node("target_query", query_target, state),
        "viewpoint_planning": lambda state: run_node("viewpoint_planning", plan_viewpoint, state),
        "navigation_execution": lambda state: run_node("navigation_execution", execute_navigation, state),
        "head_camera_search": lambda state: run_node("head_camera_search", search_with_head_camera, state),
        "local_approach": lambda state: run_node("local_approach", local_approach, state),
    }


def build_langgraph(context: ORSkillContext):
    graph = StateGraph(ORAgentState)
    nodes = _graph_nodes(context)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.set_entry_point("global_perception")
    for name in (
        "global_perception",
        "robot_localization",
        "target_query",
        "viewpoint_planning",
        "navigation_execution",
        "head_camera_search",
        "local_approach",
    ):
        graph.add_conditional_edges(name, _next_node, {key: key for key in nodes} | {None: END})
    return graph.compile()


def run_agent(args: argparse.Namespace) -> int:
    state = initial_state(
        args.instruction,
        scene=args.scene,
        room_usd=str(args.room_usd),
        max_steps=args.max_steps,
    )
    context = ORSkillContext(
        channel_id=args.channel_id,
        dds_interface=args.dds_interface,
        height=args.height,
        max_speed=args.max_speed,
        yaw_rate=args.yaw_rate,
        goal_tolerance_m=args.goal_tolerance,
        viewpoint_distance_m=args.viewpoint_distance,
        waypoint_tolerance_m=args.waypoint_tolerance,
        navigation_command_duration_s=args.navigation_command_duration,
        navigation_command_hz=args.navigation_command_hz,
        navigation_retry_limit=args.navigation_retry_limit,
        target_arrival_radius_m=args.target_arrival_radius,
        near_goal_acceptance_m=args.near_goal_acceptance,
        near_goal_stall_attempts=args.near_goal_stall_attempts,
        map_resolution_m=args.map_resolution,
        robot_radius_m=args.robot_radius,
        safety_margin_m=args.safety_margin,
        dynamic_obstacle_margin_m=args.dynamic_obstacle_margin,
        safety_horizon_s=args.safety_horizon,
        obstacle_queries=args.obstacle_queries,
        max_obstacle_detections=args.max_obstacle_detections,
        max_obstacle_cameras=args.max_obstacle_cameras,
        sim_python=args.sim_python,
        model_path=str(args.model_path),
        max_env_cameras=args.max_env_cameras,
        perception_timeout_s=args.perception_timeout,
        local_approach_duration_s=args.local_approach_duration,
        local_approach_retry_limit=args.local_approach_retry_limit,
        local_stop_area_ratio=args.local_stop_area_ratio,
        local_slow_area_ratio=args.local_slow_area_ratio,
        local_stop_height_ratio=args.local_stop_height_ratio,
        local_max_speed_mps=args.local_max_speed,
        local_min_speed_mps=args.local_min_speed,
        local_max_yaw_rate=args.local_max_yaw_rate,
        local_yaw_kp=args.local_yaw_kp,
        local_forward_center_limit=args.local_forward_center_limit,
    )
    graph = build_langgraph(context)
    started = time.monotonic()
    try:
        result = graph.invoke(state)
    finally:
        context.stop()
    elapsed = time.monotonic() - started
    print("[or-agent] result=" + json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"[or-agent] elapsed={elapsed:.2f}s")
    return 0 if result.get("status") == "done" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured-state OR navigation agent.")
    parser.add_argument("instruction")
    parser.add_argument("--scene", choices=["halo", "pulm"], default="halo")
    parser.add_argument("--room-usd", type=Path, required=True)
    parser.add_argument("--channel-id", type=int, default=0)
    parser.add_argument("--dds-interface", default="lo")
    parser.add_argument("--height", type=float, default=0.8)
    parser.add_argument("--max-speed", type=float, default=0.35)
    parser.add_argument("--yaw-rate", type=float, default=2.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--viewpoint-distance", type=float, default=0.9)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.25)
    parser.add_argument("--navigation-command-duration", type=float, default=0.8)
    parser.add_argument("--navigation-command-hz", type=float, default=20.0)
    parser.add_argument("--navigation-retry-limit", type=int, default=40)
    parser.add_argument("--target-arrival-radius", type=float, default=0.85)
    parser.add_argument("--near-goal-acceptance", type=float, default=0.55)
    parser.add_argument("--near-goal-stall-attempts", type=int, default=3)
    parser.add_argument("--map-resolution", type=float, default=0.10)
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.20)
    parser.add_argument("--dynamic-obstacle-margin", type=float, default=0.25)
    parser.add_argument("--safety-horizon", type=float, default=0.5)
    parser.add_argument("--obstacle-queries", default="", help="Optional natural-language hint; obstacle categories are inferred by VLM.")
    parser.add_argument("--max-obstacle-detections", type=int, default=7)
    parser.add_argument("--max-obstacle-cameras", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--sim-python", default="")
    parser.add_argument("--model-path", type=Path, default=Path("models/vlm/qwen3-vl-2b-instruct"))
    parser.add_argument("--max-env-cameras", type=int, default=6)
    parser.add_argument("--perception-timeout", type=float, default=120.0)
    parser.add_argument("--local-approach-duration", type=float, default=30.0)
    parser.add_argument("--local-approach-retry-limit", type=int, default=10)
    parser.add_argument("--local-stop-area-ratio", type=float, default=0.07)
    parser.add_argument("--local-slow-area-ratio", type=float, default=0.16)
    parser.add_argument("--local-stop-height-ratio", type=float, default=0.35)
    parser.add_argument("--local-max-speed", type=float, default=0.28)
    parser.add_argument("--local-min-speed", type=float, default=0.08)
    parser.add_argument("--local-max-yaw-rate", type=float, default=0.42)
    parser.add_argument("--local-yaw-kp", type=float, default=0.55)
    parser.add_argument("--local-forward-center-limit", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    return run_agent(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
