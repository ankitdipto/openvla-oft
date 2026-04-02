"""Evaluate RoboCasa-trained OpenVLA-OFT checkpoints on live RoboCasa tasks."""

import logging
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import imageio
import numpy as np

import wandb

# Keep transformers on a PyTorch-only path for this evaluation flow.
os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# When launched from the broader workspace root, the outer `robosuite/` repo
# directory can be seen by Python as a namespace package and shadow the actual
# `robosuite` Python package. Put the package root ahead of cwd to make imports
# deterministic for this evaluator.
WORKSPACE_ROOT = REPO_ROOT.parent
ROBOSUITE_PACKAGE_ROOT = WORKSPACE_ROOT / "robosuite"
if ROBOSUITE_PACKAGE_ROOT.is_dir() and str(ROBOSUITE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOSUITE_PACKAGE_ROOT))

from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets.robocasa_utils import (
    action_to_env_action,
    observation_from_env,
    task_description_from_env_obs,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
ACTIVE_LOG_FILE = None


def _write_signal_message(message: str) -> None:
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:
        pass

    global ACTIVE_LOG_FILE
    if ACTIVE_LOG_FILE is not None:
        try:
            ACTIVE_LOG_FILE.write(message + "\n")
            ACTIVE_LOG_FILE.flush()
        except Exception:
            pass


def _handle_termination_signal(signum, _frame) -> None:
    signal_name = signal.Signals(signum).name
    _write_signal_message(f"Received {signal_name}; terminating evaluation.")
    raise SystemExit(128 + signum)


for _signal in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_signal, _handle_termination_signal)


@dataclass
class GenerateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = True
    num_open_loop_steps: int = 1
    lora_rank: int = 32
    dataset_name: str = "robocasa_pickplace_counter_to_cabinet"
    unnorm_key: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # RoboCasa
    task_name: str = "PickPlaceCounterToCabinet"
    split: str = "target"
    num_trials: int = 5
    max_steps: int = 300

    # Logging
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    save_rollout_videos: bool = True

    # Repro
    seed: int = 7


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.num_open_loop_steps >= 1, "num_open_loop_steps must be >= 1"
    assert cfg.num_images_in_input in (1, 2), "RoboCasa eval currently supports 1 or 2 images"


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-robocasa-{cfg.task_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    global ACTIVE_LOG_FILE
    ACTIVE_LOG_FILE = log_file
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=cfg.__dict__,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def make_robocasa_env(cfg: GenerateConfig):
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    return RoboCasaGymEnv(env_name=cfg.task_name, split=cfg.split, enable_render=True)


def prepare_observation(obs: Dict[str, Any], resize_size: int, cfg: GenerateConfig) -> Dict[str, Any]:
    return observation_from_env(
        obs,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
        image_resize_size=resize_size,
        image_resize_fn=resize_image_for_policy,
    )


def resolve_unnorm_key(cfg: GenerateConfig, model: Any) -> str:
    requested = cfg.unnorm_key or cfg.dataset_name
    norm_stats = getattr(model, "norm_stats", None) or {}
    if requested in norm_stats:
        return requested
    if len(norm_stats) == 1:
        return next(iter(norm_stats))
    available = ", ".join(sorted(norm_stats.keys())) or "none"
    raise ValueError(
        f"Could not resolve unnorm_key `{requested}` from checkpoint statistics; available keys: {available}"
    )


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    rollout_dir = "./experiments/rollouts"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = re.sub(r"[^a-z0-9_]+", "_", task_description.lower())
    processed_task_description = processed_task_description.strip("_")[:50] or "task"
    mp4_path = f"{rollout_dir}/{DATE_TIME}.eps{idx}.mp4"
    imageio.mimwrite(mp4_path, rollout_images, fps=20)
    log_message(f"Saved rollout video to {mp4_path}", log_file)


def compose_rollout_frame(obs) -> np.ndarray:
    """Compose a side-by-side rollout frame with third-person and wrist views."""
    third_person = np.asarray(obs["video.robot0_agentview_left"], dtype=np.uint8)
    wrist = np.asarray(obs["video.robot0_eye_in_hand"], dtype=np.uint8)

    if third_person.shape[0] != wrist.shape[0]:
        raise ValueError(
            "Expected RoboCasa rollout cameras to share the same height for side-by-side video composition."
        )

    return np.concatenate((third_person, wrist), axis=1)


def run_episode(
    cfg: GenerateConfig,
    env,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    episode_idx: int = 0,
    log_file=None,
):
    obs, _ = env.reset(seed=cfg.seed + episode_idx)
    task_description = task_description_from_env_obs(obs)
    print("Task description: ", task_description)
    action_queue = deque(maxlen=max(cfg.num_open_loop_steps, NUM_ACTIONS_CHUNK))

    replay_images = []
    success = False

    for t in range(cfg.max_steps):
        observation = prepare_observation(obs, resize_size, cfg)
        replay_images.append(compose_rollout_frame(obs))

        if len(action_queue) == 0:
            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            for act in actions[: cfg.num_open_loop_steps]:
                action_queue.append(act)
            log_message(f"episode={episode_idx} t={t}: queried policy for {len(actions)} actions", log_file)

        env_action = action_to_env_action(action_queue.popleft())
        obs, reward, terminated, truncated, info = env.step(env_action)

        if info.get("success", False):
            success = True
            log_message(f"episode={episode_idx} succeeded at t={t}", log_file)
            break

        if terminated or truncated:
            log_message(f"episode={episode_idx} ended early at t={t} (terminated={terminated}, truncated={truncated})", log_file)
            break

    if not success:
        log_message(f"episode={episode_idx} finished without success", log_file)

    if cfg.save_rollout_videos:
        save_rollout_video(replay_images, episode_idx, success, task_description, log_file)

    return success, task_description


def initialize_model(cfg: GenerateConfig):
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_noisy_action_projector,
        get_processor,
        get_proprio_projector,
    )

    model = get_model(cfg)
    processor = get_processor(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=PROPRIO_DIM)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    return model, processor, action_head, proprio_projector, noisy_action_projector


@draccus.wrap() # type: ignore
def eval_robocasa(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    log_file, _, run_id = setup_logging(cfg)

    log_message(f"Run ID: {run_id}", log_file)
    log_message(f"Evaluating task `{cfg.task_name}` on RoboCasa split `{cfg.split}`", log_file)

    model, processor, action_head, proprio_projector, noisy_action_projector = initialize_model(cfg)
    cfg.unnorm_key = resolve_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)
    log_message(f"Using unnorm_key `{cfg.unnorm_key}`", log_file)

    env = make_robocasa_env(cfg)
    successes = 0
    task_description = cfg.task_name
    try:
        for episode_idx in range(cfg.num_trials):
            success, task_description = run_episode(
                cfg,
                env,
                model,
                resize_size,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                episode_idx=episode_idx,
                log_file=log_file,
            )
            successes += int(success)
            success_rate = successes / (episode_idx + 1)
            log_message(
                f"progress: {successes}/{episode_idx + 1} successful episodes (success rate {success_rate:.3f})",
                log_file,
            )

            if cfg.use_wandb:
                wandb.log(
                    {
                        "success_rate": success_rate,
                        "successes": successes,
                        "episodes": episode_idx + 1,
                    }
                )
    finally:
        env.close()
        log_file.close()
        if cfg.use_wandb:
            wandb.finish()

    final_success_rate = successes / max(cfg.num_trials, 1)
    print(f"Final success rate on `{task_description}`: {final_success_rate:.3f}")
    return final_success_rate


if __name__ == "__main__":
    eval_robocasa() # type: ignore
