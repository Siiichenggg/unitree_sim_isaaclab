# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import os

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg, CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.g1_tasks.move_cylinder_g1_29dof_dex3_wholebody import mdp


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


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[vlm-scene] invalid {name}={raw!r}, using default {default}")
        return default
    if value <= 0:
        print(f"[vlm-scene] invalid {name}={raw!r}, using default {default}")
        return default
    return value


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
_floor_collider_z_defaults = {
    "halo": 0.085,
    "pulm": 0.145,
}
room_usd_path = _room_usd_env or _room_usd_defaults[_or_scene]
robot_init_pos = _float_tuple_from_env("UNITREE_ROBOT_INIT_POS", _robot_init_pos_defaults[_or_scene])
robot_init_rot = _float_tuple_from_env("UNITREE_ROBOT_INIT_ROT", (0.7071, 0.0, 0.0, 0.7071))
floor_collider_z = float(os.environ.get("UNITREE_OR_FLOOR_COLLIDER_Z", str(_floor_collider_z_defaults[_or_scene])))
floor_collider_size = _float_tuple_from_env("UNITREE_OR_FLOOR_COLLIDER_SIZE", (7.0, 7.0, 0.04))
or_camera_width = _int_from_env("UNITREE_OR_CAMERA_WIDTH", 1920)
or_camera_height = _int_from_env("UNITREE_OR_CAMERA_HEIGHT", 1080)


def _or_head_camera_cfg(**kwargs) -> CameraCfg:
    return CameraBaseCfg.get_camera_config(
        height=or_camera_height,
        width=or_camera_width,
        **kwargs,
    )


def _room_camera_cfg(camera_name: str) -> CameraCfg:
    return CameraCfg(
        prim_path=f"/World/envs/env_.*/Room/Cameras/Blender/{camera_name}",
        update_period=0.02,
        height=or_camera_height,
        width=or_camera_width,
        data_types=["rgb"],
        spawn=None,
    )


@configclass
class ORVLMSceneCfg(InteractiveSceneCfg):
    """OR-only Dex3 wholebody scene for VLM search.

    This scene is independent of the original object-moving task scene. It
    contains only the OR room, a stable floor collider, the robot, and cameras.
    """

    room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(usd_path=room_usd_path),
    )

    floor_collider = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ORFloorCollider",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, floor_collider_z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=floor_collider_size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                opacity=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
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

    front_camera = _or_head_camera_cfg()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()

    if _or_scene == "halo":
        scene_camera_00 = _room_camera_cfg("Camera_40795974")
        scene_camera_01 = _room_camera_cfg("Camera_40034694")
        scene_camera_02 = _room_camera_cfg("Camera_46026258")
        scene_camera_03 = _room_camera_cfg("Camera_47661457")
        scene_camera_04 = _room_camera_cfg("PoV_Camera_0")
        scene_camera_05 = _room_camera_cfg("PoV_Camera_1")
        scene_camera_06 = _room_camera_cfg("PoV_Camera_2")
        scene_camera_07 = _room_camera_cfg("PoV_Camera_3")
        scene_camera_08 = _room_camera_cfg("PoV_Camera_4")
        scene_camera_09 = _room_camera_cfg("iso")
        scene_camera_10 = _room_camera_cfg("RobotCam")
    else:
        scene_camera_00 = _room_camera_cfg("Camera_45902703L")
        scene_camera_01 = _room_camera_cfg("Camera_46517772L")
        scene_camera_02 = _room_camera_cfg("Camera_44664489L")
        scene_camera_03 = _room_camera_cfg("Camera_41908851L")
        scene_camera_04 = _room_camera_cfg("optimized")


@configclass
class ActionsCfg:
    """Wholebody joint-position action configuration."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation configuration used by the DDS/VLM control stack."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class EventCfg:
    pass


@configclass
class ORVLMG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """G1 Dex3 wholebody VLM environment in the OR room."""

    scene: ORVLMSceneCfg = ORVLMSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards = None
    curriculum = None

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
