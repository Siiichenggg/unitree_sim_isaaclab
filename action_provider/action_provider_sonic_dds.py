# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from typing import Optional
import json
import math
import time
import logging

import torch

from action_provider.action_base import ActionProvider
from dds.dds_master import dds_manager


UNITREE_G1_LOW_CMD_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

SONIC_G1_DEFAULT_ANGLES = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
]

SONIC_G1_EFFORT_LIMITS = [
    88.0,
    88.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    88.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
]


class SonicDDSActionProvider(ActionProvider):
    """Bridge SONIC C++ deploy LowCmd targets into IsaacLab joint targets."""

    def __init__(self, env, args_cli):
        super().__init__("SonicDDSActionProvider")
        self.env = env
        self.args_cli = args_cli
        self.device = env.device
        self.robot = env.scene["robot"]

        if args_cli.robot_type != "g129":
            raise RuntimeError("sonic_dds currently supports robot_type='g129' only")

        self.robot_dds = dds_manager.get_object("g129")
        if self.robot_dds is None:
            raise RuntimeError("g129 DDS object not found. Did create_dds_objects() run?")

        self.joint_names = list(self.robot.data.joint_names)
        self.joint_to_index = {name: i for i, name in enumerate(self.joint_names)}
        missing = [name for name in UNITREE_G1_LOW_CMD_ORDER if name not in self.joint_to_index]
        if missing:
            raise RuntimeError(f"Robot asset missing G1 joints required by sonic_dds: {missing}")

        self.lowcmd_src_idx = torch.arange(len(UNITREE_G1_LOW_CMD_ORDER), device=self.device, dtype=torch.long)
        self.env_dst_idx = torch.tensor(
            [self.joint_to_index[name] for name in UNITREE_G1_LOW_CMD_ORDER],
            device=self.device,
            dtype=torch.long,
        )

        default_joint_pos = self.robot.data.default_joint_pos
        default_joint_pos = default_joint_pos.clone() if default_joint_pos.dim() == 2 else default_joint_pos.unsqueeze(0).clone()
        self.default_joint_pos = default_joint_pos.to(device=self.device, dtype=torch.float32)
        sonic_default_body_q = torch.tensor(SONIC_G1_DEFAULT_ANGLES, device=self.device, dtype=torch.float32)
        sonic_default_body_q = sonic_default_body_q.unsqueeze(0).expand(self.default_joint_pos.shape[0], -1)
        self.default_joint_pos.index_copy_(1, self.env_dst_idx, sonic_default_body_q)
        self.default_joint_vel = torch.zeros_like(self.default_joint_pos)
        self.default_q = self.default_joint_pos[0].clone()
        self.full_target = self.default_q.clone()
        self.full_velocity_target = torch.zeros_like(self.default_q)
        self.full_effort_target = torch.zeros_like(self.default_q)
        self.effort_target_batch = torch.zeros_like(self.default_joint_pos)
        self.release_start_q = self.default_q.clone()
        self.body_kp = torch.zeros(len(UNITREE_G1_LOW_CMD_ORDER), device=self.device, dtype=torch.float32)
        self.body_kd = torch.zeros_like(self.body_kp)
        self.body_effort_limits = torch.tensor(
            SONIC_G1_EFFORT_LIMITS,
            device=self.device,
            dtype=torch.float32,
        )
        self.torque_control = bool(getattr(args_cli, "sonic_torque_control", False))
        self._gains_written = False

        root_state = getattr(self.robot.data, "root_state_w", None)
        if root_state is None:
            root_state = self.robot.data.default_root_state
        self.lock_root_pose = root_state[:, :7].clone().to(device=self.device, dtype=torch.float32)
        root_z_override = getattr(args_cli, "sonic_root_z", None)
        if root_z_override is not None:
            self.lock_root_pose[:, 2] = float(root_z_override)
        self.zero_root_velocity = torch.zeros((self.lock_root_pose.shape[0], 6), device=self.device, dtype=torch.float32)
        self.initial_root_xy = self.lock_root_pose[0, :2].clone()
        self.release_root_xy = self.initial_root_xy.clone()
        self.idle_xy_anchor = self.lock_root_pose[:, :2].clone()
        self.waiting_for_sonic = True
        self.release_start_time = None
        self.release_ramp_sec = float(getattr(args_cli, "sonic_release_ramp_sec", 1.0))
        self.release_delay_sec = float(getattr(args_cli, "sonic_release_delay_sec", 3.5))
        self.state_log_interval_sec = float(getattr(args_cli, "sonic_state_log_interval_sec", 1.0))
        self._last_state_log_time = 0.0
        self.elastic_band_enabled = bool(getattr(args_cli, "sonic_elastic_band", False))
        self.elastic_band_body_id = None
        if self.elastic_band_enabled:
            body_names = list(getattr(self.robot.data, "body_names", []))
            requested_body = str(getattr(args_cli, "sonic_elastic_band_body", "pelvis"))
            fallback_bodies = [requested_body, "pelvis", "torso_link", "base_link"]
            for body_name in fallback_bodies:
                if body_name in body_names:
                    self.elastic_band_body_id = torch.tensor([body_names.index(body_name)], device=self.device, dtype=torch.long)
                    self.elastic_band_body_name = body_name
                    break
            if self.elastic_band_body_id is None:
                self.elastic_band_enabled = False
                print(f"[{self.name}] SONIC elastic band disabled; no matching body in {fallback_bodies}")
            else:
                env_count = self.default_joint_pos.shape[0]
                self.elastic_force = torch.zeros((env_count, 1, 3), device=self.device, dtype=torch.float32)
                self.elastic_torque = torch.zeros_like(self.elastic_force)
                self.elastic_world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).expand(env_count, 3)
                elastic_height = getattr(args_cli, "sonic_elastic_band_height", None)
                if elastic_height is None:
                    elastic_height = float(self.lock_root_pose[0, 2].item())
                self.elastic_height = float(elastic_height)
                self.elastic_kp = float(getattr(args_cli, "sonic_elastic_band_kp", 10000.0))
                self.elastic_kd = float(getattr(args_cli, "sonic_elastic_band_kd", 1000.0))
                self.elastic_ang_kp = float(getattr(args_cli, "sonic_elastic_band_ang_kp", 400.0))
                self.elastic_ang_kd = float(getattr(args_cli, "sonic_elastic_band_ang_kd", 20.0))
                self.elastic_max_force = float(getattr(args_cli, "sonic_elastic_band_max_force", 5000.0))
                self.elastic_max_torque = float(getattr(args_cli, "sonic_elastic_band_max_torque", 1200.0))

        self.motion_state_file = str(getattr(args_cli, "sonic_motion_state_file", "") or "").strip()
        self.require_motion_started = bool(getattr(args_cli, "sonic_require_motion_started", False))
        self.idle_xy_lock_enabled = bool(getattr(args_cli, "sonic_idle_xy_lock", False))
        if self.idle_xy_lock_enabled and not self.elastic_band_enabled:
            self.idle_xy_lock_enabled = False
            print(f"[{self.name}] SONIC idle XY lock disabled; it requires --sonic_elastic_band")
        self.idle_xy_kp = float(getattr(args_cli, "sonic_idle_xy_kp", 1800.0))
        self.idle_xy_kd = float(getattr(args_cli, "sonic_idle_xy_kd", 360.0))
        self.idle_xy_max_force = float(getattr(args_cli, "sonic_idle_xy_max_force", 900.0))
        self.idle_xy_deadband = max(0.0, float(getattr(args_cli, "sonic_idle_xy_deadband", 0.015)))
        self.idle_xy_relock_delay_sec = max(0.0, float(getattr(args_cli, "sonic_idle_xy_relock_delay_sec", 0.75)))
        self.motion_state_stale_sec = max(0.05, float(getattr(args_cli, "sonic_motion_state_stale_sec", 0.75)))
        self._motion_state = {
            "started": False,
            "moving": False,
            "mode": None,
            "wall_time": 0.0,
            "last_move_wall_time": 0.0,
        }
        self._last_motion_poll_time = 0.0
        self._last_idle_xy_active = False
        self._idle_xy_active = False
        self._reported_waiting_motion_start = False

        self.has_soft_limits = hasattr(self.robot.data, "soft_joint_pos_limits")
        if self.has_soft_limits:
            limits = self.robot.data.soft_joint_pos_limits
            limits = limits[0] if limits.dim() == 3 else limits
            limits = limits.to(device=self.device, dtype=torch.float32)
            self.lower_limits = limits[:, 0]
            self.upper_limits = limits[:, 1]

        physics_dt = getattr(args_cli, "physics_dt", None)
        if physics_dt is None:
            physics_dt = getattr(env, "physics_dt", None)
        if physics_dt is None:
            physics_dt = getattr(env.sim, "dt", 0.005)
        self.physics_dt = float(physics_dt)
        self.control_dt = 1.0 / float(getattr(args_cli, "step_hz", 50) or 50)
        self.substeps = max(1, round(self.control_dt / self.physics_dt))
        self.observation_dt = self.physics_dt * self.substeps
        setattr(self.env, "_sonic_observation_dt", self.observation_dt)
        self.render_interval = max(1, int(getattr(args_cli, "render_interval", 1) or 1))
        self._render_counter = 0

        self.start_time = time.perf_counter()
        self.last_cmd_time = 0.0
        self.last_cmd_stamp = None
        self.first_cmd_time = None
        self._reported_first_cmd = False
        self._reported_release_delay = False
        self._last_cmd_motor_count = 0
        self.init_hold_sec = 1.0
        self.cmd_timeout_sec = 0.20

        print(
            f"[{self.name}] initialized: {len(UNITREE_G1_LOW_CMD_ORDER)} G1 body joints, "
            f"{self.substeps} sim substeps per control step, observation_dt={self.observation_dt:.4f}s, "
            f"render_interval={self.render_interval}"
        )
        if root_z_override is not None:
            print(f"[{self.name}] root lock z override: {float(root_z_override):.3f}m")
        print(f"[{self.name}] waiting for SONIC LowCmd; root lock is active")
        if self.elastic_band_enabled:
            print(
                f"[{self.name}] SONIC elastic band enabled on body '{self.elastic_band_body_name}' "
                f"(height={self.elastic_height:.2f}, kp={self.elastic_kp:.0f}, kd={self.elastic_kd:.0f}, "
                f"ang_kp={self.elastic_ang_kp:.0f}, ang_kd={self.elastic_ang_kd:.0f})"
            )
        if self.idle_xy_lock_enabled:
            print(
                f"[{self.name}] SONIC idle XY lock enabled "
                f"(kp={self.idle_xy_kp:.0f}, kd={self.idle_xy_kd:.0f}, max_force={self.idle_xy_max_force:.0f}, "
                f"relock_delay={self.idle_xy_relock_delay_sec:.2f}s, state_file='{self.motion_state_file}')"
            )
        if self.require_motion_started:
            print(f"[{self.name}] root release gated on keyboard teleop started=true")

    def _update_body_from_lowcmd(self, cmd_data) -> bool:
        if not cmd_data or "motor_cmd" not in cmd_data:
            return False

        motor_cmd = cmd_data["motor_cmd"]
        positions = motor_cmd.get("positions", [])
        if len(positions) < len(UNITREE_G1_LOW_CMD_ORDER):
            return False

        q_hw = torch.as_tensor(
            positions[: len(UNITREE_G1_LOW_CMD_ORDER)],
            dtype=torch.float32,
            device=self.device,
        )
        q_targets = q_hw.index_select(0, self.lowcmd_src_idx)
        self.full_target.index_copy_(0, self.env_dst_idx, q_targets)

        velocities = motor_cmd.get("velocities", [])
        if len(velocities) >= len(UNITREE_G1_LOW_CMD_ORDER):
            dq_hw = torch.as_tensor(
                velocities[: len(UNITREE_G1_LOW_CMD_ORDER)],
                dtype=torch.float32,
                device=self.device,
            )
            self.full_velocity_target.index_copy_(0, self.env_dst_idx, dq_hw.index_select(0, self.lowcmd_src_idx))
        else:
            self.full_velocity_target.zero_()

        torques = motor_cmd.get("torques", [])
        if len(torques) >= len(UNITREE_G1_LOW_CMD_ORDER):
            tau_hw = torch.as_tensor(
                torques[: len(UNITREE_G1_LOW_CMD_ORDER)],
                dtype=torch.float32,
                device=self.device,
            )
            self.full_effort_target.index_copy_(0, self.env_dst_idx, tau_hw.index_select(0, self.lowcmd_src_idx))
        else:
            self.full_effort_target.zero_()

        kp = motor_cmd.get("kp", [])
        kd = motor_cmd.get("kd", [])
        if len(kp) >= len(UNITREE_G1_LOW_CMD_ORDER) and len(kd) >= len(UNITREE_G1_LOW_CMD_ORDER):
            kp_hw = torch.as_tensor(
                kp[: len(UNITREE_G1_LOW_CMD_ORDER)],
                dtype=torch.float32,
                device=self.device,
            )
            kd_hw = torch.as_tensor(
                kd[: len(UNITREE_G1_LOW_CMD_ORDER)],
                dtype=torch.float32,
                device=self.device,
            )
            self.body_kp.copy_(kp_hw.index_select(0, self.lowcmd_src_idx))
            self.body_kd.copy_(kd_hw.index_select(0, self.lowcmd_src_idx))

        if self.has_soft_limits:
            self.full_target.copy_(torch.maximum(self.full_target, self.lower_limits))
            self.full_target.copy_(torch.minimum(self.full_target, self.upper_limits))

        receive_time = cmd_data.get("_receive_time")
        if receive_time is not None:
            self.last_cmd_time = float(receive_time)
        else:
            cmd_stamp = cmd_data.get("_timestamp")
            if cmd_stamp != self.last_cmd_stamp or self.last_cmd_time <= 0.0:
                self.last_cmd_time = time.perf_counter()
                self.last_cmd_stamp = cmd_stamp

        if not self._reported_first_cmd:
            self._last_cmd_motor_count = len(positions)

        return True

    def _fallback_hold(self):
        self.full_target.copy_(self.default_q)
        self.full_velocity_target.zero_()
        self.full_effort_target.zero_()

    def _apply_sonic_drive_gains(self):
        if self._gains_written or torch.count_nonzero(self.body_kp).item() == 0:
            return
        env_count = self.default_joint_pos.shape[0]
        kp = self.body_kp.unsqueeze(0).expand(env_count, -1)
        kd = self.body_kd.unsqueeze(0).expand(env_count, -1)
        effort_limits = self.body_effort_limits.unsqueeze(0).expand(env_count, -1)
        if self.torque_control:
            self.robot.write_joint_stiffness_to_sim(torch.zeros_like(kp), joint_ids=self.env_dst_idx)
            self.robot.write_joint_damping_to_sim(torch.zeros_like(kd), joint_ids=self.env_dst_idx)
        else:
            self.robot.write_joint_stiffness_to_sim(kp, joint_ids=self.env_dst_idx)
            self.robot.write_joint_damping_to_sim(kd, joint_ids=self.env_dst_idx)
        self.robot.write_joint_effort_limit_to_sim(effort_limits, joint_ids=self.env_dst_idx)
        self._gains_written = True
        print(
            f"[{self.name}] applied SONIC LowCmd drive gains "
            f"(kp {self.body_kp.min().item():.1f}-{self.body_kp.max().item():.1f}, "
            f"kd {self.body_kd.min().item():.2f}-{self.body_kd.max().item():.2f}, "
            f"effort {self.body_effort_limits.min().item():.1f}-{self.body_effort_limits.max().item():.1f}, "
            f"mode={'explicit_torque' if self.torque_control else 'implicit_position'})"
        )

    def _apply_explicit_pd_torque(self, target):
        target_batch = target.unsqueeze(0).expand(self.default_joint_pos.shape[0], -1)
        velocity_batch = self.full_velocity_target.unsqueeze(0).expand_as(target_batch)
        feedforward_batch = self.full_effort_target.unsqueeze(0).expand_as(target_batch)
        q = self.robot.data.joint_pos[:, self.env_dst_idx]
        dq = self.robot.data.joint_vel[:, self.env_dst_idx]
        kp = self.body_kp.unsqueeze(0)
        kd = self.body_kd.unsqueeze(0)
        limits = self.body_effort_limits.unsqueeze(0)
        tau = feedforward_batch[:, self.env_dst_idx] + kp * (target_batch[:, self.env_dst_idx] - q) + kd * (
            velocity_batch[:, self.env_dst_idx] - dq
        )
        tau = torch.clamp(tau, -limits, limits)
        self.effort_target_batch.zero_()
        self.effort_target_batch.index_copy_(1, self.env_dst_idx, tau)
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.set_joint_velocity_target(self.default_joint_vel)
        self.robot.set_joint_effort_target(self.effort_target_batch)

    def _capture_current_root_lock(self):
        root_state = getattr(self.robot.data, "root_state_w", None)
        if root_state is not None:
            self.lock_root_pose = root_state[:, :7].clone().to(device=self.device, dtype=torch.float32)
            self.release_root_xy = self.lock_root_pose[0, :2].clone()
            self.idle_xy_anchor = self.lock_root_pose[:, :2].clone()
            self.zero_root_velocity = torch.zeros(
                (self.lock_root_pose.shape[0], 6),
                device=self.device,
                dtype=torch.float32,
            )

    def _lock_root_pose_only(self):
        self.robot.write_root_pose_to_sim(self.lock_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)

    def _hold_default_joints(self):
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.set_joint_velocity_target(self.default_joint_vel)
        self.robot.set_joint_effort_target(self.default_joint_vel)

    def _apply_joint_targets(self, target):
        if self.torque_control:
            self._apply_explicit_pd_torque(target)
        else:
            self.robot.set_joint_position_target(target)
            self.robot.set_joint_velocity_target(self.full_velocity_target)
            self.robot.set_joint_effort_target(self.full_effort_target)

    def _read_motion_state(self, now):
        if not self.motion_state_file:
            return self._motion_state
        if now - self._last_motion_poll_time < 0.05:
            return self._motion_state
        self._last_motion_poll_time = now
        try:
            with open(self.motion_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return self._motion_state

        self._motion_state = {
            "started": bool(state.get("started", False)),
            "moving": bool(state.get("moving", False)),
            "mode": state.get("mode"),
            "wall_time": float(state.get("wall_time", 0.0) or 0.0),
            "last_move_wall_time": float(state.get("last_move_wall_time", 0.0) or 0.0),
        }
        return self._motion_state

    def _set_idle_xy_anchor_from_current_body(self):
        if self.elastic_band_enabled and self.elastic_band_body_id is not None:
            body_id = int(self.elastic_band_body_id[0].item())
            self.idle_xy_anchor = self.robot.data.body_link_pose_w[:, body_id, :2].clone()
            return
        root_state = getattr(self.robot.data, "root_state_w", None)
        if root_state is not None:
            self.idle_xy_anchor = root_state[:, :2].clone()

    def _motion_state_started_recent(self, now):
        if not self.require_motion_started:
            return True
        state = self._read_motion_state(now)
        wall_time = float(state.get("wall_time", 0.0) or 0.0)
        if wall_time <= 0.0 or time.time() - wall_time > self.motion_state_stale_sec:
            return False
        return bool(state.get("started", False))

    def _update_idle_xy_lock_state(self, now):
        if not self.idle_xy_lock_enabled or self.waiting_for_sonic:
            active = False
        else:
            state = self._read_motion_state(now)
            wall_time = float(state.get("wall_time", 0.0) or 0.0)
            stale = wall_time <= 0.0 or time.time() - wall_time > self.motion_state_stale_sec
            try:
                mode = int(state.get("mode"))
            except (TypeError, ValueError):
                mode = -1
            last_move_wall_time = float(state.get("last_move_wall_time", 0.0) or 0.0)
            recently_moved = (
                last_move_wall_time > 0.0
                and time.time() - last_move_wall_time < self.idle_xy_relock_delay_sec
            )
            active = (
                bool(state.get("started", False))
                and not bool(state.get("moving", False))
                and mode == 0
                and not stale
                and not recently_moved
            )

        if active and not self._last_idle_xy_active:
            self._set_idle_xy_anchor_from_current_body()
            xy = self.idle_xy_anchor[0]
            print(f"[{self.name}] idle XY lock active at ({xy[0].item():.3f}, {xy[1].item():.3f})")
        elif not active and self._last_idle_xy_active:
            print(f"[{self.name}] idle XY lock released")
        self._last_idle_xy_active = active
        self._idle_xy_active = active
        return active

    def _quat_body_z_axis_w(self, quat_wxyz):
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        return torch.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=1,
        )

    def _apply_elastic_band(self):
        if not self.elastic_band_enabled:
            return

        body_id = int(self.elastic_band_body_id[0].item())
        body_pose = self.robot.data.body_link_pose_w[:, body_id, :]
        body_vel = self.robot.data.body_link_vel_w[:, body_id, :]
        pos = body_pose[:, :3]
        quat = body_pose[:, 3:7]
        lin_vel = body_vel[:, :3]
        ang_vel = body_vel[:, 3:6]

        self.elastic_force.zero_()
        self.elastic_torque.zero_()
        if self._idle_xy_active:
            error_xy = self.idle_xy_anchor - pos[:, :2]
            if self.idle_xy_deadband > 0.0:
                error_sign = torch.sign(error_xy)
                error_mag = torch.clamp(torch.abs(error_xy) - self.idle_xy_deadband, min=0.0)
                error_xy = error_sign * error_mag
            force_xy = self.idle_xy_kp * error_xy - self.idle_xy_kd * lin_vel[:, :2]
            self.elastic_force[:, 0, :2] = torch.clamp(force_xy, -self.idle_xy_max_force, self.idle_xy_max_force)
        force_z = self.elastic_kp * (self.elastic_height - pos[:, 2]) - self.elastic_kd * lin_vel[:, 2]
        self.elastic_force[:, 0, 2] = torch.clamp(force_z, -self.elastic_max_force, self.elastic_max_force)

        if self.elastic_ang_kp > 0.0 or self.elastic_ang_kd > 0.0:
            body_z_w = self._quat_body_z_axis_w(quat)
            upright_axis = torch.cross(body_z_w, self.elastic_world_up, dim=1)
            torque = self.elastic_ang_kp * upright_axis - self.elastic_ang_kd * ang_vel
            self.elastic_torque[:, 0, :] = torch.clamp(torque, -self.elastic_max_torque, self.elastic_max_torque)

        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            self.robot.set_external_force_and_torque(
                self.elastic_force,
                self.elastic_torque,
                body_ids=self.elastic_band_body_id,
                is_global=True,
            )
        finally:
            logging.disable(previous_disable_level)

    def _pre_release_target(self, now):
        if self.first_cmd_time is None or self.release_delay_sec <= 0.0:
            return self.full_target
        alpha = min(1.0, max(0.0, (now - self.first_cmd_time) / self.release_delay_sec))
        return self.default_q.lerp(self.full_target, alpha)

    def _log_root_stability(self, now):
        if self.state_log_interval_sec <= 0.0 or self.waiting_for_sonic:
            return
        if now - self._last_state_log_time < self.state_log_interval_sec:
            return
        self._last_state_log_time = now
        root_state = getattr(self.robot.data, "root_state_w", None)
        if root_state is None or root_state.numel() == 0:
            return
        root_z = float(root_state[0, 2].item())
        root_x = float(root_state[0, 0].item())
        root_y = float(root_state[0, 1].item())
        dx_release = float((root_state[0, 0] - self.release_root_xy[0]).item())
        dy_release = float((root_state[0, 1] - self.release_root_xy[1]).item())
        q = root_state[0, 3:7]
        w, x, y, _z = [float(v.item()) for v in q]
        body_up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
        tilt_deg = math.degrees(math.acos(body_up_z))
        max_joint_vel = float(torch.max(torch.abs(self.robot.data.joint_vel[0, self.env_dst_idx])).item())
        fall_suspect = root_z < 0.55 or tilt_deg > 50.0
        print(
            f"[{self.name}] state root=({root_x:.3f},{root_y:.3f},{root_z:.3f})m "
            f"d_release=({dx_release:.3f},{dy_release:.3f})m tilt={tilt_deg:.1f}deg "
            f"max_joint_vel={max_joint_vel:.2f}rad/s idle_xy_lock={self._idle_xy_active} "
            f"fall_suspect={fall_suspect}"
        )

    def get_action(self, env) -> Optional[torch.Tensor]:
        try:
            now = time.perf_counter()
            cmd_data = self.robot_dds.get_robot_command()
            got_cmd = self._update_body_from_lowcmd(cmd_data)
            timed_out = self.last_cmd_time > 0.0 and now - self.last_cmd_time > self.cmd_timeout_sec
            command_active = got_cmd and not timed_out

            if command_active:
                if not self._reported_first_cmd:
                    self.first_cmd_time = now
                    print(f"[{self.name}] received first active SONIC LowCmd with {self._last_cmd_motor_count} motors")
                    if self.release_delay_sec > 0.0:
                        print(f"[{self.name}] keeping root locked for {self.release_delay_sec:.2f}s to let SONIC InitControl finish")
                    self._reported_first_cmd = True
                self._apply_sonic_drive_gains()

                release_delay_done = (
                    self.first_cmd_time is None
                    or self.release_delay_sec <= 0.0
                    or now - self.first_cmd_time >= self.release_delay_sec
                )
                motion_start_done = self._motion_state_started_recent(now)
                if release_delay_done and not motion_start_done and not self._reported_waiting_motion_start:
                    print(f"[{self.name}] LowCmd ready; waiting for keyboard teleop start before root release")
                    self._reported_waiting_motion_start = True
                if self.waiting_for_sonic and release_delay_done:
                    if not motion_start_done:
                        release_delay_done = False
                if self.waiting_for_sonic and release_delay_done:
                    current_q = self.robot.data.joint_pos[0].clone()
                    max_jump = torch.max(torch.abs(self.full_target - current_q)).item()
                    print(f"[{self.name}] SONIC target max delta from current pose at root release: {max_jump:.3f} rad")
                    print(f"[{self.name}] SONIC LowCmd active; releasing root lock")
                    self.release_start_q.copy_(current_q)
                    root_state = getattr(self.robot.data, "root_state_w", None)
                    if root_state is not None and root_state.numel() > 0:
                        self.release_root_xy = root_state[0, :2].clone()
                    self._set_idle_xy_anchor_from_current_body()
                    self.release_start_time = now
                    self.waiting_for_sonic = False
            else:
                if timed_out and not self.waiting_for_sonic:
                    print(f"[{self.name}] SONIC LowCmd timeout; locking robot until commands resume")
                    self._capture_current_root_lock()
                    self.waiting_for_sonic = True
                    self.first_cmd_time = None
                    self._reported_first_cmd = False
                self._fallback_hold()

            for _ in range(self.substeps):
                self._update_idle_xy_lock_state(now)
                if self.waiting_for_sonic:
                    self._lock_root_pose_only()
                    if command_active:
                        self._apply_joint_targets(self._pre_release_target(now))
                    else:
                        self._hold_default_joints()
                else:
                    target = self.full_target
                    if self.release_start_time is not None and self.release_ramp_sec > 0:
                        alpha = min(1.0, (now - self.release_start_time) / self.release_ramp_sec)
                        if alpha < 1.0:
                            target = self.release_start_q.lerp(self.full_target, alpha)
                        else:
                            self.release_start_time = None
                    self._apply_joint_targets(target)
                self._apply_elastic_band()
                self.env.scene.write_data_to_sim()
                self.env.sim.step(render=False)
                self.env.scene.update(dt=self.physics_dt)

            if not getattr(self.args_cli, "no_render", False):
                self._render_counter += 1
                if self._render_counter % self.render_interval == 0:
                    self.env.sim.render()

            self.env.observation_manager.compute()
            self._log_root_stability(now)
            return None

        except Exception as e:
            print(f"[{self.name}] failed: {e}")
            return None

    def cleanup(self):
        try:
            if self.robot_dds:
                self.robot_dds.stop_communication()
        except Exception as e:
            print(f"[{self.name}] Clean up DDS resources failed: {e}")
