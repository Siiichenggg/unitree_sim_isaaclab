# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import math
import os

import torch
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg, CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.g1_tasks.move_cylinder_g1_29dof_dex3_wholebody.move_cylinder_g1_29dof_dex3_hw_env_cfg import (
    MoveCylinderG129Dex3WholebodyEnvCfg,
)


def _float_tuple_from_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        values = tuple(float(v.strip()) for v in raw.replace(",", " ").split())
    except ValueError:
        print(f"[vlm-scene] invalid {name}={raw!r}, using default {default}")
        return default
    if len(values) != len(default):
        print(f"[vlm-scene] invalid {name} length {len(values)}, expected {len(default)}; using default {default}")
        return default
    return values


def _quat_multiply(q1: tuple, q2: tuple) -> tuple:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _normalize_quat(q: tuple) -> tuple:
    norm = math.sqrt(sum(component * component for component in q))
    return tuple(component / norm for component in q)


def _head_camera_pitch(name: str, pitch_deg: float):
    half = math.radians(pitch_deg) * 0.5
    pitch_quat = (math.cos(half), math.sin(half), 0.0, 0.0)
    return CameraBaseCfg.get_camera_config(
        prim_path=f"/World/envs/env_.*/Robot/d435_link/{name}",
        rot_offset=_normalize_quat(_quat_multiply((0.5, -0.5, 0.5, -0.5), pitch_quat)),
    )


def _head_camera_yaw(name: str, yaw_deg: float):
    half = math.radians(yaw_deg) * 0.5
    yaw_quat = (math.cos(half), 0.0, math.sin(half), 0.0)
    return CameraBaseCfg.get_camera_config(
        prim_path=f"/World/envs/env_.*/Robot/d435_link/{name}",
        rot_offset=_normalize_quat(_quat_multiply((0.5, -0.5, 0.5, -0.5), yaw_quat)),
    )


project_root = os.environ.get("PROJECT_ROOT") or os.getcwd()
_room_usd_env = os.environ.get("UNITREE_ROOM_USD")
_or_scene = os.environ.get("UNITREE_OR_SCENE", "").strip().lower()
if not _or_scene and _room_usd_env and "pulm" in _room_usd_env.lower():
    _or_scene = "pulm"
if _or_scene not in {"halo", "pulm"}:
    _or_scene = "halo"

_room_usd_defaults = {
    "halo": f"{project_root}/assets/objects/OR/Model/halo_room_baked/halo_room_baked.usd",
    "pulm": f"{project_root}/assets/objects/OR/Model/pulm_room_baked/pulm_room_baked.usd",
}
_robot_init_pos_defaults = {
    "halo": (1.35, -1.45, 0.91),
    "pulm": (2.1, -1.2, 0.98),
}

room_usd_path = _room_usd_env or _room_usd_defaults[_or_scene]
robot_init_pos = _float_tuple_from_env("UNITREE_ROBOT_INIT_POS", _robot_init_pos_defaults[_or_scene])
robot_init_rot = _float_tuple_from_env("UNITREE_ROBOT_INIT_ROT", (0.7071, 0.0, 0.0, 0.7071))


@configclass
class VLMORSceneCfg(InteractiveSceneCfg):
    """OR-only Dex3 wholebody scene for VLM search.

    This scene intentionally does not inherit the base move-cylinder scene,
    because that base scene brings packing-table assets and task lights that
    are already represented by the self-contained OR room USD.
    """

    room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(usd_path=room_usd_path),
    )

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=robot_init_pos,
        init_rot=robot_init_rot,
    )
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()
    front_camera_up = _head_camera_pitch("front_cam_up", pitch_deg=20.0)
    front_camera_down = _head_camera_pitch("front_cam_down", pitch_deg=-20.0)
    front_camera_left = _head_camera_yaw("front_cam_left", yaw_deg=-25.0)
    front_camera_right = _head_camera_yaw("front_cam_right", yaw_deg=25.0)


@configclass
class MoveCylinderG129Dex3WholebodyVLMEnvCfg(MoveCylinderG129Dex3WholebodyEnvCfg):
    """VLM-specific Dex3 wholebody task using an OR-only scene."""

    scene: VLMORSceneCfg = VLMORSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    rewards = None

    def __post_init__(self):
        """Post initialization without object-specific reset/reward dependencies."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()

        def reset_all(env):
            base_mdp.reset_scene_to_default(env, torch.arange(env.num_envs, device=env.device))

        self.event_manager.register("reset_all_self", SimpleEvent(func=reset_all))
        self.event_manager.register("reset_object_self", SimpleEvent(func=reset_all))
