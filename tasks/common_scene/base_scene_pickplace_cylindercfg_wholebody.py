# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")
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
    # OR baked room floors are above world z=0 at the robot spawn points.
    # Keep the G1's flat-ground root clearance (~0.80 m) relative to the visible room floor.
    "halo": (1.35, -1.45, 0.91),
    "pulm": (2.1, -1.2, 0.98),
}


def _float_from_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[scene] invalid {name}={raw!r}, using default {default}")
        return default


def _float_tuple_from_env(name, default, expected_len):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        values = tuple(float(v.strip()) for v in raw.replace(",", " ").split())
    except ValueError:
        print(f"[scene] invalid {name}={raw!r}, using default {default}")
        return default
    if len(values) != expected_len:
        print(f"[scene] invalid {name} length {len(values)}, expected {expected_len}; using default {default}")
        return default
    return values


room_usd_path = _room_usd_env or _room_usd_defaults[_or_scene]
robot_init_pos = _float_tuple_from_env("UNITREE_ROBOT_INIT_POS", _robot_init_pos_defaults[_or_scene], 3)
robot_init_rot = _float_tuple_from_env("UNITREE_ROBOT_INIT_ROT", (0.7071, 0.0, 0.0, 0.7071), 4)
or_light_pos = _float_tuple_from_env("UNITREE_OR_LIGHT_POS", (robot_init_pos[0], robot_init_pos[1], 2.35), 3)
or_light_intensity = _float_from_env("UNITREE_OR_LIGHT_INTENSITY", 3500.0)
or_light_radius = _float_from_env("UNITREE_OR_LIGHT_RADIUS", 1.2)
@configclass
class TableCylinderSceneCfgWH(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(usd_path=room_usd_path),
    )


        # 1. table configuration
    packing_table1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_1",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-2.35644,-3.45572,-0.2],   # initial position [x, y, z]
                                                rot=[0.70091, 0.0, 0.0, 0.71325]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable_2/PackingTable.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )

    packing_table2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_2",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-3.97225,-4.3424,-0.2],   # initial position [x, y, z]
                                                rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )
    # # Object
    # 2. object configuration (cylinder)     
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",    # object in the scene
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.58514,-2.78975,0.84], # initial position (pos) 
                                                  rot=[1, 0, 0, 0]), # initial rotation (rot)
        spawn=sim_utils.CylinderCfg(
            radius=0.018,    # cylinder radius (radius)
            height=0.35,     # cylinder height (height)
 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),    # rigid body properties configuration (rigid_props)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
            collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),    # visual material configuration (visual_material)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",    # friction combine mode
                restitution_combine_mode="min",    # restitution combine mode
                static_friction=1.5,    # static friction coefficient
                dynamic_friction=1.5,    # dynamic friction coefficient
                restitution=0.0,    # restitution coefficient (no restitution)
            ),
        ),
    )

    # Ground plane


    # Lights: keep the OR USD baked textures visible with one lightweight indoor source.
    light = AssetBaseCfg(
        prim_path="/World/ORInteriorLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=or_light_pos),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.96, 0.90),
            intensity=or_light_intensity,
            radius=or_light_radius,
        ),
    )
    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-0.1, 3.6, 1.6),
                                                    rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                    focal_length=16.5)
