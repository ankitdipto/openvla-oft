"""Native LeRobot-backed dataset utilities for RoboCasa/OpenVLA-OFT fine-tuning."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Type

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.vla.action_tokenizer import ActionTokenizer
if TYPE_CHECKING:
    from prismatic.models.backbones.llm.prompting import PromptBuilder

from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    NormalizationType,
)
from prismatic.vla.datasets.robocasa_utils import (
    PRIMARY_IMAGE_KEY,
    WRIST_IMAGE_KEY,
    extract_action,
    extract_proprio,
)


@dataclass
class LeRobotBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Callable[[Image.Image], torch.Tensor]
    prompt_builder_fn: Type["PromptBuilder"]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = sample["dataset_name"]
        actions = sample["action"]
        current_action = actions[0]
        future_actions = actions[1:]

        prompt_builder = self.prompt_builder_fn("openvla")
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        lang = sample["task"]["language_instruction"].lower()
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(sample["observation"]["image_primary"])

        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            "actions": actions,
        }
        if self.use_wrist_image and sample["observation"].get("image_wrist") is not None:
            output["pixel_values_wrist"] = self.image_transform(sample["observation"]["image_wrist"])
        if self.use_proprio and sample["observation"].get("proprio") is not None:
            output["proprio"] = sample["observation"]["proprio"]
        return output

class LeRobotDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str,
        batch_transform: LeRobotBatchTransform,
        shuffle_buffer_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform
        self.train = train
        self.image_aug = image_aug
        self.use_wrist_image = batch_transform.use_wrist_image
        self.use_proprio = batch_transform.use_proprio
        self.shuffle_buffer_size = shuffle_buffer_size

        if not self.data_root_dir.exists():
            raise FileNotFoundError(f"LeRobot dataset root does not exist: {self.data_root_dir}")

        self.info = self._load_json(self.data_root_dir / "meta" / "info.json")
        self.modality = self._load_json(self.data_root_dir / "meta" / "modality.json")
        self.raw_stats = self._load_json(self.data_root_dir / "meta" / "stats.json")
        self.task_text_by_index = self._load_task_texts(self.data_root_dir / "meta" / "tasks.jsonl")
        self.episode_records = self._load_episodes(self.data_root_dir / "meta" / "episodes.jsonl")
        self.episode_records_by_index = {entry["episode_index"]: entry for entry in self.episode_records}
        self.episode_indices = self._resolve_split_indices(self.train)

        self.action_slices = self.modality["action"]
        self.state_slices = self.modality["state"]
        self.video_path_template = self.info["video_path"]
        self.data_path_template = self.info["data_path"]

        self.dataset_length = sum(
            max(0, self.episode_records_by_index[idx]["length"] - NUM_ACTIONS_CHUNK + 1) for idx in self.episode_indices
        )
        self.dataset_statistics = self._build_dataset_statistics()

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_jsonl(path: Path) -> List[dict]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_task_texts(self, path: Path) -> Dict[int, str]:
        return {entry["task_index"]: entry["task"] for entry in self._load_jsonl(path)}

    def _load_episodes(self, path: Path) -> List[dict]:
        episodes = self._load_jsonl(path)
        episodes.sort(key=lambda entry: entry["episode_index"])
        return episodes

    def _resolve_split_indices(self, train: bool) -> List[int]:
        split_name = "train" if train else "val"
        splits = self.info.get("splits", {})
        if split_name in splits:
            start, end = [int(x) for x in splits[split_name].split(":")]
            return list(range(start, end))

        if train and self.episode_records:
            return [entry["episode_index"] for entry in self.episode_records]

        available_splits = ", ".join(sorted(splits)) or "none"
        raise ValueError(
            f"LeRobot dataset at {self.data_root_dir} does not define a '{split_name}' split; available splits: {available_splits}"
        )

    def _episode_chunk_dir(self, episode_index: int) -> int:
        return episode_index // int(self.info["chunks_size"])

    def _episode_parquet_path(self, episode_index: int) -> Path:
        rel_path = self.data_path_template.format(
            episode_chunk=self._episode_chunk_dir(episode_index), episode_index=episode_index
        )
        return self.data_root_dir / rel_path

    def _episode_video_path(self, episode_index: int, video_key: str) -> Path:
        rel_path = self.video_path_template.format(
            episode_chunk=self._episode_chunk_dir(episode_index),
            episode_index=episode_index,
            video_key=video_key,
        )
        return self.data_root_dir / rel_path

    def _extract_action(self, action_array: np.ndarray) -> np.ndarray:
        return extract_action(action_array, self.action_slices)

    def _extract_proprio(self, state_array: np.ndarray) -> np.ndarray:
        return extract_proprio(state_array, self.state_slices)

    def _subset_stats(self, stats: Dict[str, List[float]], indices: List[int]) -> Dict[str, np.ndarray]:
        return {name: np.asarray(values, dtype=np.float32)[indices] for name, values in stats.items()}

    def _compute_stats(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "mean": values.mean(axis=0).astype(np.float32),
            "std": values.std(axis=0).astype(np.float32),
            "min": values.min(axis=0).astype(np.float32),
            "max": values.max(axis=0).astype(np.float32),
            "q01": np.quantile(values, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(values, 0.99, axis=0).astype(np.float32),
        }

    def _build_dataset_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        action_indices = []
        for key in ("end_effector_position", "end_effector_rotation", "gripper_close"):
            action_indices.extend(range(self.action_slices[key]["start"], self.action_slices[key]["end"]))
        stats = {
            self.dataset_name: {
                "action": self._subset_stats(self.raw_stats["action"], action_indices),
                "num_trajectories": len(self.episode_indices),
                "num_transitions": sum(self.episode_records_by_index[idx]["length"] for idx in self.episode_indices),
            }
        }
        if self.use_proprio:
            all_proprio = []
            for episode_index in self.episode_indices:
                df = pd.read_parquet(self._episode_parquet_path(episode_index), columns=["observation.state"])
                states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
                all_proprio.append(self._extract_proprio(states))
            stats[self.dataset_name]["proprio"] = self._compute_stats(np.concatenate(all_proprio, axis=0))
        return stats

    def _normalize(self, values: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.NORMAL:
            return (values - stats["mean"]) / (stats["std"] + 1e-8)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            low, high = stats["min"], stats["max"]
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            low, high = stats["q01"], stats["q99"]
        else:
            raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

        normalized = 2 * (values - low) / (high - low + 1e-8) - 1
        normalized = np.clip(normalized, -1.0, 1.0)
        zeros_mask = stats["min"] == stats["max"]
        normalized[..., zeros_mask] = 0.0
        return normalized.astype(np.float32)

    def _maybe_augment(self, image: Image.Image) -> Image.Image:
        return image

    def _read_frame(self, capture: cv2.VideoCapture) -> Image.Image:
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Failed to read frame from RoboCasa LeRobot video.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _iter_episode(self, episode_index: int):
        df = pd.read_parquet(
            self._episode_parquet_path(episode_index),
            columns=["action", "observation.state", "task_index"],
        )
        actions_full = np.stack(df["action"].to_numpy()).astype(np.float32)
        task_indices = df["task_index"].to_numpy()
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32) if self.use_proprio else None
        usable_steps = max(0, len(actions_full) - NUM_ACTIONS_CHUNK + 1)
        if usable_steps <= 0:
            return

        primary_cap = cv2.VideoCapture(str(self._episode_video_path(episode_index, PRIMARY_IMAGE_KEY)))
        wrist_cap = None
        if self.use_wrist_image:
            wrist_cap = cv2.VideoCapture(str(self._episode_video_path(episode_index, WRIST_IMAGE_KEY)))

        try:
            for step_idx in range(usable_steps):
                primary_image = self._maybe_augment(self._read_frame(primary_cap))
                wrist_image = self._maybe_augment(self._read_frame(wrist_cap)) if wrist_cap is not None else None

                action_chunk = self._extract_action(actions_full[step_idx : step_idx + NUM_ACTIONS_CHUNK])
                normalized_actions = self._normalize(action_chunk, self.dataset_statistics[self.dataset_name]["action"])

                observation = {"image_primary": primary_image}
                if wrist_image is not None:
                    observation["image_wrist"] = wrist_image
                if self.use_proprio and states is not None:
                    proprio = self._extract_proprio(states[step_idx : step_idx + 1])
                    observation["proprio"] = self._normalize(
                        proprio[0], self.dataset_statistics[self.dataset_name]["proprio"]
                    ).astype(np.float32)

                task_index = int(task_indices[step_idx])
                sample = {
                    "dataset_name": self.dataset_name,
                    "action": normalized_actions,
                    "observation": observation,
                    "task": {"language_instruction": self.task_text_by_index[task_index]},
                }
                yield self.batch_transform(sample)
        finally:
            primary_cap.release()
            if wrist_cap is not None:
                wrist_cap.release()

    def __iter__(self):
        episode_indices = list(self.episode_indices)
        while True:
            if self.train:
                random.shuffle(episode_indices)
            for episode_index in episode_indices:
                yield from self._iter_episode(episode_index)
            if not self.train:
                break

    def __len__(self) -> int:
        return self.dataset_length
