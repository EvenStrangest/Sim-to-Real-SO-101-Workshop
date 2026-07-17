# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Return the state of the end effector frame in the robot coordinate system.
    """
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w, robot.data.root_quat_w
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos, ee_frame_quat = ee_frame.data.target_pos_w[:, 0, :], ee_frame.data.target_quat_w[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    ee_frame_state = torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)

    return ee_frame_state


def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:

    sensor = env.scene[sensor_cfg.name]
    images = sensor.data.output[data_type]

    return images.clone()


# ---------------------------------------------------------------------------
# Oracle 3-D state terms (oracle-3d-sensing branch).
#
# Raw world-frame poses of the task objects, the end-effector frame, and the
# robot base, recorded per step for the sparse-object-3-D channel (terraforge
# VIALS_TO_RACK_SIM2REAL_DESIGN_AND_PLAN.md §6: record RAW poses, derive
# keypoints offline). Poses are env-origin-relative world frame; quaternions
# are Isaac (w, x, y, z). One term serves demo collection AND live eval.
# ---------------------------------------------------------------------------


def oracle_object_poses_w(
    env: ManagerBasedRLEnv,
    object_names: list[str],
) -> torch.Tensor:
    """Concatenated [pos(3), quat_wxyz(4)] per object, world frame relative to
    the env origin. Shape (E, 7 * len(object_names))."""
    origins = env.scene.env_origins
    chunks = []
    for name in object_names:
        obj = env.scene[name]
        pos = obj.data.root_pos_w - origins
        quat = obj.data.root_quat_w
        chunks.append(torch.cat([pos, quat], dim=1))
    return torch.cat(chunks, dim=1)


def oracle_ee_pose_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector (gripper-link) pose [pos(3), quat_wxyz(4)] in env-origin-
    relative world frame. Shape (E, 7)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins
    quat = ee_frame.data.target_quat_w[:, 0, :]
    return torch.cat([pos, quat], dim=1)


def oracle_robot_base_pose_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot base pose [pos(3), quat_wxyz(4)] in env-origin-relative world
    frame — recorded so the world->robot-base re-expression of the 3-D channel
    is self-contained offline. Shape (E, 7)."""
    robot: Articulation = env.scene[robot_cfg.name]
    pos = robot.data.root_pos_w - env.scene.env_origins
    quat = robot.data.root_quat_w
    return torch.cat([pos, quat], dim=1)