"""Shared RoboCasa data-contract utilities for training and evaluation."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


PRIMARY_IMAGE_KEY = "observation.images.robot0_agentview_left"
WRIST_IMAGE_KEY = "observation.images.robot0_eye_in_hand"
TASK_DESCRIPTION_KEY = "annotation.human.task_description"

ACTION_KEY_ORDER = ("end_effector_position", "end_effector_rotation", "gripper_close")
STATE_KEY_ORDER = (
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float32, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = float(np.sqrt(max(1.0 - quat[3] * quat[3], 0.0)))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def extract_action(action_array: np.ndarray, action_slices: Dict[str, Dict[str, int]]) -> np.ndarray:
    chunks = []
    for key in ACTION_KEY_ORDER:
        start, end = action_slices[key]["start"], action_slices[key]["end"]
        chunks.append(action_array[:, start:end])
    return np.concatenate(chunks, axis=-1).astype(np.float32)


def extract_proprio(state_array: np.ndarray, state_slices: Dict[str, Dict[str, int]]) -> np.ndarray:
    pos = state_array[
        :,
        state_slices["end_effector_position_relative"]["start"] : state_slices["end_effector_position_relative"]["end"],
    ]
    quat = state_array[
        :,
        state_slices["end_effector_rotation_relative"]["start"] : state_slices["end_effector_rotation_relative"]["end"],
    ]
    axis_angle = np.stack([quat2axisangle(q) for q in quat], axis=0)
    gripper = state_array[:, state_slices["gripper_qpos"]["start"] : state_slices["gripper_qpos"]["end"]]
    return np.concatenate((pos, axis_angle, gripper), axis=-1).astype(np.float32)


def task_description_from_env_obs(obs: Dict[str, Any]) -> str:
    return str(obs[TASK_DESCRIPTION_KEY])


def observation_from_env(
    obs: Dict[str, Any],
    *,
    use_wrist_image: bool,
    use_proprio: bool,
    image_resize_size: int | None = None,
    image_resize_fn=None,
) -> Dict[str, Any]:
    primary = np.asarray(obs["video.robot0_agentview_left"], dtype=np.uint8)
    if image_resize_size is not None and image_resize_fn is not None:
        primary = image_resize_fn(primary, image_resize_size)

    output: Dict[str, Any] = {"full_image": primary}

    if use_wrist_image:
        wrist = np.asarray(obs["video.robot0_eye_in_hand"], dtype=np.uint8)
        if image_resize_size is not None and image_resize_fn is not None:
            wrist = image_resize_fn(wrist, image_resize_size)
        output["wrist_image"] = wrist

    if use_proprio:
        proprio = np.concatenate(
            (
                np.asarray(obs["state.end_effector_position_relative"], dtype=np.float32),
                quat2axisangle(np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float32)),
                np.asarray(obs["state.gripper_qpos"], dtype=np.float32),
            )
        ).astype(np.float32)
        output["state"] = proprio

    return output


def action_to_env_action(action: np.ndarray) -> Dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    pos = np.clip(action[:3], -1.0, 1.0)
    rot = np.clip(action[3:6], -1.0, 1.0)
    gripper = np.clip(action[6:7], 0.0, 1.0)
    return {
        "action.end_effector_position": pos,
        "action.end_effector_rotation": rot,
        "action.gripper_close": gripper,
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.zeros(1, dtype=np.float32),
    }
