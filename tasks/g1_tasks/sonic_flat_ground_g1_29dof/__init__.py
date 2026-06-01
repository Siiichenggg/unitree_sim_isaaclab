# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import gymnasium as gym

from . import sonic_flat_ground_env_cfg


gym.register(
    id="Isaac-Sonic-FlatGround-G129-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_flat_ground_env_cfg.SonicFlatGroundG129WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
