# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import or_vlm_g1_29dof_dex3_wholebody_env_cfg


gym.register(
    id="Isaac-OR-VLM-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": or_vlm_g1_29dof_dex3_wholebody_env_cfg.ORVLMG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
