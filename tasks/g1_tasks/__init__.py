
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

from importlib import import_module

_TASK_MODULES = [
        "pick_place_cylinder_g1_29dof_dex3", "pick_place_cylinder_g1_29dof_dex1", 
        "pick_place_redblock_g1_29dof_dex1", "pick_place_redblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_dex1", "stack_rgyblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_inspire",
        "pick_redblock_into_drawer_g1_29dof_dex1","pick_redblock_into_drawer_g1_29dof_dex3",
        "pick_place_redblock_g1_29dof_inspire",
        "pick_place_cylinder_g1_29dof_inspire",
        "move_cylinder_g1_29dof_dex1_wholebody",
        "move_cylinder_g1_29dof_dex3_wholebody",
        "or_vlm_g1_29dof_dex3_wholebody",
        "move_cylinder_g1_29dof_inspire_wholebody",
        "sonic_flat_ground_g1_29dof",
        "or_halo_room_g1_29dof_dex1",
        "or_pulm_room_g1_29dof_dex1",
]

__all__ = []

for _module_name in _TASK_MODULES:
    try:
        globals()[_module_name] = import_module(f"{__name__}.{_module_name}")
        __all__.append(_module_name)
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{_module_name}":
            raise
        print(f"[tasks.g1_tasks] optional task module missing, skipped: {_module_name}")
