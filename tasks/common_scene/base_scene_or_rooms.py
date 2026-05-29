# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Common operating-room scene configurations."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OR_MODEL_DIR = PROJECT_ROOT / "assets" / "objects" / "OR" / "Model"

OR_ROOM_SCALE = (1.0, 1.0, 1.0)
OR_ROOM_ROT = [1.0, 0.0, 0.0, 0.0]
OR_ROOM_POS = [0.0, 0.0, 0.0]


@configclass
class HaloOperatingRoomSceneCfg(InteractiveSceneCfg):
    """Operating-room scene backed by the halo room USD."""

    operating_room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/OperatingRoom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=OR_ROOM_POS,
            rot=OR_ROOM_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OR_MODEL_DIR / "halo_room_baked" / "halo_room_baked.usd"),
            scale=OR_ROOM_SCALE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    ambient_light = AssetBaseCfg(
        prim_path="/World/ambient_light",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=800.0, visible_in_primary_ray=False),
    )

    robot_key_light = AssetBaseCfg(
        prim_path="/World/robot_key_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.5, 3.0)),
        spawn=sim_utils.SphereLightCfg(color=(1.0, 0.96, 0.9), intensity=12000.0, radius=1.25),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-0.1, 3.6, 1.6),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )


@configclass
class PulmOperatingRoomSceneCfg(InteractiveSceneCfg):
    """Operating-room scene backed by the pulmonary room USD."""

    operating_room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/OperatingRoom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=OR_ROOM_POS,
            rot=OR_ROOM_ROT,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OR_MODEL_DIR / "pulm_room_baked" / "pulm_room_baked.usd"),
            scale=OR_ROOM_SCALE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    ambient_light = AssetBaseCfg(
        prim_path="/World/ambient_light",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=800.0, visible_in_primary_ray=False),
    )

    robot_key_light = AssetBaseCfg(
        prim_path="/World/robot_key_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.5, 3.0)),
        spawn=sim_utils.SphereLightCfg(color=(1.0, 0.96, 0.9), intensity=12000.0, radius=1.25),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-0.1, 3.6, 1.6),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )
