# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg
from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import robot_init_pos
from tasks.g1_tasks.move_cylinder_g1_29dof_dex3_wholebody.move_cylinder_g1_29dof_dex3_hw_env_cfg import (
    MoveCylinderG129Dex3WholebodyEnvCfg,
    ObjectTableSceneCfg,
)


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[vlm-lighting] invalid {name}={raw!r}, using default {default}")
        return default


def _float_tuple_from_env(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        values = tuple(float(v.strip()) for v in raw.replace(",", " ").split())
    except ValueError:
        print(f"[vlm-lighting] invalid {name}={raw!r}, using default {default}")
        return default
    if len(values) != 3:
        print(f"[vlm-lighting] invalid {name} length {len(values)}, expected 3; using default {default}")
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


vlm_ambient_intensity = _float_from_env("UNITREE_VLM_AMBIENT_INTENSITY", 800.0)
vlm_robot_fill_pos = _float_tuple_from_env(
    "UNITREE_VLM_ROBOT_FILL_POS",
    (robot_init_pos[0], robot_init_pos[1], robot_init_pos[2] + 0.75),
)
vlm_robot_fill_intensity = _float_from_env("UNITREE_VLM_ROBOT_FILL_INTENSITY", 0.0)
vlm_robot_fill_radius = _float_from_env("UNITREE_VLM_ROBOT_FILL_RADIUS", 1.8)


@configclass
class VLMObjectTableSceneCfg(ObjectTableSceneCfg):
    """Dex3 wholebody scene with extra fixed head-camera views for VLM search."""

    vlm_ambient_light = AssetBaseCfg(
        prim_path="/World/VLMAmbientFillLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.92, 0.96, 1.0),
            intensity=vlm_ambient_intensity,
            visible_in_primary_ray=False,
        ),
    )
    vlm_robot_fill_light = AssetBaseCfg(
        prim_path="/World/VLMRobotFillLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=vlm_robot_fill_pos),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.97, 0.92),
            intensity=vlm_robot_fill_intensity,
            radius=vlm_robot_fill_radius,
        ),
    )
    front_camera_up = _head_camera_pitch("front_cam_up", pitch_deg=20.0)
    front_camera_down = _head_camera_pitch("front_cam_down", pitch_deg=-20.0)
    front_camera_left = _head_camera_yaw("front_cam_left", yaw_deg=-25.0)
    front_camera_right = _head_camera_yaw("front_cam_right", yaw_deg=25.0)


@configclass
class MoveCylinderG129Dex3WholebodyVLMEnvCfg(MoveCylinderG129Dex3WholebodyEnvCfg):
    """VLM-specific Dex3 wholebody task that leaves the base Dex3 task unchanged."""

    scene: VLMObjectTableSceneCfg = VLMObjectTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
