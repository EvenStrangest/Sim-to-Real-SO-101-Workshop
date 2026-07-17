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
#
# --- oracle-3d-sensing branch (terraforge perceived-3-D workstream) ---
#
# INSTRUMENTED variants of the vials-to-rack task for the sparse-object-3-D
# workstream (terraforge VIALS_TO_RACK_SIM2REAL_DESIGN_AND_PLAN.md §6 +
# THREE_D_OBB_SENSING_DESIGN_AND_PLAN.md sensing-stage 0). They add, ON TOP of
# the stock task (which stays byte-identical for baseline parity):
#
#   1. an `oracle` observation group — raw world-frame poses of
#      [vial_1, vial_2, vial_3, rack_left], the EE frame, and the robot base
#      (keypoints are derived OFFLINE by the shared canonical-OBB library);
#   2. a FIXED dedicated geometry camera (`camera_geometry`) — the sim analog
#      of the physical decoupled RealSense D405: RGB-D + instance seg, never
#      part of the policy observation, never randomized (the external-camera
#      pose randomization moves LightBox/camera_mount, which this camera is
#      deliberately NOT parented to).
#
# COMPATIBILITY NOTE: the stock scripts (lerobot_eval.py / lerobot_agent.py)
# auto-discover every scene entity named `camera_*` and forward ALL of them to
# the policy — so these instrumented variants MUST NOT be used with the stock
# scripts unless the caller restricts the interface's camera dict to the
# policy cameras (ego + external_D455). The terraforge drivers do exactly that.

import numpy as np

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101.mdp import (
    oracle_object_poses_w,
    oracle_ee_pose_w,
    oracle_robot_base_pose_w,
)

from .task_env_cfg import camera_object
from .vials_to_rack_env_cfg import (
    VialsToRackSceneCfg,
    VialsToRackEventCfg,
    VialsToRackObservationsCfg,
    VialsToRackTerminationsCfg,
    VialsToRackEnvCfg,
)

# Fixed roster + slot order of the sparse-object-3-D channel (gap 20).
ORACLE_ROSTER = ["vial_1", "vial_2", "vial_3", "rack_left"]

# Geometry camera placement (gap 29): fixed in the env frame, oblique view of
# the whole mat workspace from the -y side. Chosen 2026-07-17: position
# (0.22, -0.30, 0.35) looking at ~(0.20, 0, 0.05) -> standoff ~0.42 m to the
# workspace center (0.33-0.48 m across the object region — inside a D405-ish
# close-range envelope). Same pinhole model as the other sim cameras (D405
# intrinsic realism is deliberately deferred — plan gap 33).
GEOMETRY_CAMERA_POS = (0.22, -0.30, 0.35)
# Extrinsic-XYZ euler (deg), opengl convention: Rx(+45.07) pitches the
# straight-down identity view up to ~44.9 deg below horizontal toward +y;
# Rz(+3.8) turns the heading to the workspace center.
GEOMETRY_CAMERA_EULER_DEG = (45.07, 0.0, 3.8)

camera_geometry_cfg = camera_object.replace()
camera_geometry_cfg.prim_path = "{ENV_REGEX_NS}/geometry_cam"
camera_geometry_cfg.offset.pos = GEOMETRY_CAMERA_POS
camera_geometry_cfg.offset.rot = euler_angles_to_quat(
    np.array(GEOMETRY_CAMERA_EULER_DEG), degrees=True
)


@configclass
class VialsToRackOracleSceneCfg(VialsToRackSceneCfg):
    """Stock vials-to-rack scene + the fixed geometry camera."""

    camera_geometry = camera_geometry_cfg


@configclass
class VialsToRackOracleObservationsCfg(VialsToRackObservationsCfg):
    """Stock observations + the oracle group."""

    @configclass
    class OracleCfg(ObsGroup):
        object_poses_w = ObsTerm(
            func=oracle_object_poses_w,
            params={"object_names": ORACLE_ROSTER},
        )
        ee_pose_w = ObsTerm(
            func=oracle_ee_pose_w,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        robot_base_pose_w = ObsTerm(
            func=oracle_robot_base_pose_w,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    oracle: OracleCfg = OracleCfg()


@configclass
class VialsToRackOracleEnvCfg(VialsToRackEnvCfg):
    """Instrumented vials-to-rack (teleop/recording: no terminations)."""

    scene: VialsToRackOracleSceneCfg = VialsToRackOracleSceneCfg()
    events: VialsToRackEventCfg = VialsToRackEventCfg()
    observations: VialsToRackOracleObservationsCfg = VialsToRackOracleObservationsCfg()


@configclass
class VialsToRackOracleEvalEnvCfg(VialsToRackOracleEnvCfg):
    """Instrumented vials-to-rack with the stock Eval terminations."""

    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0


# Self-registering (this module is imported by tasks/__init__.py's
# import_packages sweep).
import gymnasium as gym  # noqa: E402

gym.register(
    id="Lerobot-So101-Teleop-Vials-To-Rack-Oracle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:VialsToRackOracleEnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-Teleop-Vials-To-Rack-Oracle-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:VialsToRackOracleEvalEnvCfg",
    },
)
