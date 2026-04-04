from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import os
import re
import shutil
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
from peft import PeftModel

# Keep helper utilities on the PyTorch-only Transformers path. This env may
# intentionally omit or break TensorFlow because the RoboCasa workflow here does
# not depend on the TFDS/RLDS stack.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoModelForVision2Seq
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
from prismatic.vla.datasets.statistics import save_dataset_statistics

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_backend: str = "rlds"                       # Dataset backend to use (`rlds` or `lerobot`)
    data_root_dir: Path = Path("datasets/rlds")      # Dataset root (RLDS root dir or LeRobot dataset dir)
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Dataset identifier / statistics key used during fine-tuning
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10 percent to 100 percent)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
    overwrite_prev_checkpoint_after_first: bool = True  # If True, keep the first checkpoint and overwrite one rolling
                                                         #   later checkpoint on subsequent saves
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging is slow and best avoided for routine checkpoints.
                                                     #         Leave this False for fast resumable checkpoints and merge
                                                     #         final checkpoints offline when needed.

    # Logging
    wandb_entity: str = "ankitdipto"          # Name of WandB entity
    wandb_project: str = "VLA-RL"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps
    log_step_timing: bool = False                    # If True, prints wall-clock timing for each training iteration

    # fmt: on


def get_base_model_path_metadata_path(checkpoint_dir: Path) -> Path:
    """Return the metadata file used to persist the original base-model path for adapter-only checkpoints."""
    return checkpoint_dir / "base_model_path.txt"


def load_base_model_path_from_checkpoint(checkpoint_dir: str | Path) -> Optional[str]:
    """Load the original base-model path recorded alongside an adapter-only checkpoint."""
    metadata_path = get_base_model_path_metadata_path(Path(checkpoint_dir))
    if not metadata_path.exists():
        return None
    base_model_path = metadata_path.read_text().strip()
    return base_model_path or None


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        latest_checkpoint_path = os.path.join(path, f"{module_name}--latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint_path):
            checkpoint_path = latest_checkpoint_path
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def infer_resume_step(checkpoint_dir: str) -> int:
    """
    Infer the training step for a checkpoint directory.

    Supports explicit `--STEP_chkpt` directory names, latest-checkpoint state
    files, and directories containing multiple trainer state files.
    """
    checkpoint_path = Path(checkpoint_dir)
    assert checkpoint_path.is_dir(), f"Resume checkpoint path must be a directory: {checkpoint_dir}"

    latest_state_path = checkpoint_path / "trainer_state--latest_checkpoint.pt"
    if latest_state_path.exists():
        trainer_state = torch.load(latest_state_path, weights_only=False, map_location="cpu")
        return int(trainer_state["step"])

    match = re.search(r"--(\d+)_chkpt$", checkpoint_path.name)
    if match:
        return int(match.group(1))

    candidate_steps = []
    for trainer_state_path in checkpoint_path.glob("trainer_state--*_checkpoint.pt"):
        step_match = re.search(r"trainer_state--(\d+)_checkpoint\.pt$", trainer_state_path.name)
        if step_match:
            candidate_steps.append(int(step_match.group(1)))

    assert candidate_steps, f"Could not infer resume step from checkpoint directory: {checkpoint_dir}"
    return max(candidate_steps)


def load_trainer_state(path: str, step: int, device: str = "cpu") -> dict:
    """Load optimizer, scheduler, and metadata for a checkpoint."""
    trainer_state_path = os.path.join(path, f"trainer_state--{step}_checkpoint.pt")
    if not os.path.exists(trainer_state_path):
        latest_state_path = os.path.join(path, "trainer_state--latest_checkpoint.pt")
        if os.path.exists(latest_state_path):
            trainer_state_path = latest_state_path

    print(f"Loading trainer state: {trainer_state_path}")
    return torch.load(trainer_state_path, weights_only=False, map_location=device)


def move_optimizer_state_to_device(optimizer: AdamW, device_id: int) -> None:
    """Move tensor optimizer state onto the active device after loading."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def override_optimizer_and_scheduler_lr(optimizer: AdamW, scheduler: _LRScheduler, learning_rate: float) -> None:
    """Force the resumed optimizer/scheduler to use the CLI-provided learning rate."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["initial_lr"] = learning_rate

    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [learning_rate for _ in scheduler.base_lrs]
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [learning_rate for _ in scheduler._last_lr]

def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def get_wandb_run_id_path(run_dir: Path) -> Path:
    """Return the file path used to persist the W&B run id for resumable logging."""
    return run_dir / "wandb_run_id.txt"


def load_wandb_run_id(run_dir: Path) -> Optional[str]:
    """Load a previously persisted W&B run id if one exists."""
    path = get_wandb_run_id_path(run_dir)
    if not path.exists():
        return None
    run_id = path.read_text().strip()
    return run_id or None


def save_wandb_run_id(run_dir: Path, wandb_run_id: str) -> None:
    """Persist the active W&B run id alongside the experiment artifacts."""
    get_wandb_run_id_path(run_dir).write_text(wandb_run_id.strip() + "\n")


def load_latest_local_wandb_step(wandb_root: Path, wandb_run_id: str) -> Optional[int]:
    """Return the maximum locally cached W&B `_step` for a given run id, if available."""
    if not wandb_run_id:
        return None

    summary_steps = []
    for summary_path in wandb_root.glob(f"run-*-{wandb_run_id}/files/wandb-summary.json"):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        step = summary.get("_step")
        if step is not None:
            try:
                summary_steps.append(int(step))
            except (TypeError, ValueError):
                continue

    if not summary_steps:
        return None
    return max(summary_steps)





def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def _get_latest_checkpoint_symlink_path(run_dir: Path) -> Path:
    """Return the stable symlink path for the latest completed checkpoint."""
    return Path(str(run_dir) + "--latest_chkpt")


def _get_numbered_checkpoint_dir(run_dir: Path, log_step: int) -> Path:
    """Return the numbered checkpoint directory for a given training step."""
    return Path(str(run_dir) + f"--{log_step}_chkpt")


def _get_temp_checkpoint_dir(final_checkpoint_dir: Path) -> Path:
    """Return the temporary directory used while assembling a checkpoint."""
    return Path(str(final_checkpoint_dir) + ".tmp")


def _get_latest_checkpoint_target(run_dir: Path) -> Optional[Path]:
    """Return the previous completed checkpoint pointed to by the latest symlink, if any."""
    latest_symlink = _get_latest_checkpoint_symlink_path(run_dir)
    if not latest_symlink.is_symlink():
        return None

    target = os.readlink(latest_symlink)
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = latest_symlink.parent / target_path
    return target_path.resolve()


def _update_latest_checkpoint_symlink(run_dir: Path, final_checkpoint_dir: Path) -> Optional[Path]:
    """
    Atomically point the stable latest-checkpoint symlink to the newest completed numbered checkpoint.

    This keeps the previous latest checkpoint intact until the new checkpoint directory is fully written.
    """
    latest_symlink = _get_latest_checkpoint_symlink_path(run_dir)
    previous_checkpoint_dir = _get_latest_checkpoint_target(run_dir)

    # Preserve any legacy real directory from the old overwrite-in-place scheme.
    if latest_symlink.exists() and not latest_symlink.is_symlink():
        legacy_path = latest_symlink.with_name(latest_symlink.name + ".legacy")
        suffix = 1
        while legacy_path.exists():
            legacy_path = latest_symlink.with_name(latest_symlink.name + f".legacy.{suffix}")
            suffix += 1
        os.replace(latest_symlink, legacy_path)

    tmp_symlink = latest_symlink.with_name(latest_symlink.name + ".tmp-link")
    if tmp_symlink.exists() or tmp_symlink.is_symlink():
        tmp_symlink.unlink()

    target_name = os.path.basename(final_checkpoint_dir)
    os.symlink(target_name, tmp_symlink)
    os.replace(tmp_symlink, latest_symlink)
    return previous_checkpoint_dir


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    base_model_path,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    optimizer,
    scheduler,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Always save into a fresh numbered checkpoint directory first.
    # This avoids corrupting the last good checkpoint if the job is terminated during save.
    final_checkpoint_dir = _get_numbered_checkpoint_dir(run_dir, log_step)
    temp_checkpoint_dir = _get_temp_checkpoint_dir(final_checkpoint_dir)
    checkpoint_name_suffix = "latest_checkpoint.pt"
    adapter_dir = temp_checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        if temp_checkpoint_dir.exists():
            shutil.rmtree(temp_checkpoint_dir)
        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, temp_checkpoint_dir)
        get_base_model_path_metadata_path(temp_checkpoint_dir).write_text(str(base_model_path).strip() + "\n")
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(temp_checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(
                proprio_projector.state_dict(), temp_checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(),
                temp_checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}",
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), temp_checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(),
                temp_checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}",
            )

        torch.save(
            {
                "step": log_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            temp_checkpoint_dir / f"trainer_state--{checkpoint_name_suffix}",
        )

    # Wait for model components to be saved
    dist.barrier()

    # Finalize checkpoint directory and atomically advance the latest symlink.
    if distributed_state.is_main_process:
        if cfg.use_lora and cfg.merge_lora_during_training:
            base_vla = AutoModelForVision2Seq.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
            merged_vla = merged_vla.merge_and_unload()
            merged_vla.save_pretrained(temp_checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {temp_checkpoint_dir}")

        if final_checkpoint_dir.exists():
            shutil.rmtree(final_checkpoint_dir)
        os.replace(temp_checkpoint_dir, final_checkpoint_dir)
        previous_checkpoint_dir = _update_latest_checkpoint_symlink(run_dir, final_checkpoint_dir)
        if previous_checkpoint_dir is not None and previous_checkpoint_dir != final_checkpoint_dir:
            shutil.rmtree(previous_checkpoint_dir, ignore_errors=True)
        print(f"Checkpoint for Step {log_step} finalized at: {final_checkpoint_dir}")

    dist.barrier()
