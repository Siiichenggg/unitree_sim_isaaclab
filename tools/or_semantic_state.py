#!/usr/bin/env python3

"""Structured semantic state for OR navigation agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict


AgentPhase = Literal[
    "global_perception",
    "robot_localization",
    "target_query",
    "viewpoint_planning",
    "navigation_execution",
    "head_camera_search",
    "local_approach",
    "done",
    "failed",
]


@dataclass
class SemanticObject:
    object_id: str
    label: str
    confidence: float = 0.0
    seen_by: list[str] = field(default_factory=list)
    bbox_by_view: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    world_xy: tuple[float, float] | None = None
    head_visible: bool = False
    reachable: bool = True
    last_seen_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ORAgentState(TypedDict, total=False):
    instruction: str
    scene: str
    room_usd: str
    phase: AgentPhase
    status: str
    step_count: int
    max_steps: int
    robot_pose: tuple[float, float, float] | None
    semantic_objects: list[dict]
    obstacle_objects: list[dict]
    occupancy_map: dict
    target_label: str
    target_object: dict | None
    nav_goal_xy: tuple[float, float] | None
    planned_path: list[tuple[float, float]]
    active_waypoint_index: int
    navigation_attempts: int
    last_nav_goal_distance: float | None
    last_nav_command: dict
    rejected_viewpoints: list[dict]
    local_approach_attempts: int
    head_visible: bool
    last_action: str
    errors: list[str]


def normalize_label(text: str) -> str:
    return " ".join(text.lower().strip().split())


def initial_state(
    instruction: str,
    *,
    scene: str,
    room_usd: str,
    max_steps: int = 12,
) -> ORAgentState:
    return {
        "instruction": instruction,
        "scene": scene,
        "room_usd": room_usd,
        "phase": "global_perception",
        "status": "running",
        "step_count": 0,
        "max_steps": max_steps,
        "robot_pose": None,
        "semantic_objects": [],
        "obstacle_objects": [],
        "occupancy_map": {},
        "target_label": normalize_label(instruction),
        "target_object": None,
        "nav_goal_xy": None,
        "planned_path": [],
        "active_waypoint_index": 0,
        "navigation_attempts": 0,
        "last_nav_goal_distance": None,
        "last_nav_command": {},
        "rejected_viewpoints": [],
        "local_approach_attempts": 0,
        "head_visible": False,
        "last_action": "",
        "errors": [],
    }
