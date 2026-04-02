# RoboCasa Workflow

This document describes the RoboCasa training and evaluation workflow in `openvla-oft`.

For environment creation, repo layout, editable installs, macros, and asset download, see:

- [ROBOCASA_SETUP.md](./ROBOCASA_SETUP.md)

## Scope

The RoboCasa integration in this repo is designed for:

- RoboCasa fine-tuning through the LeRobot backend
- resumable adapter-only checkpoints
- offline LoRA merge for export
- evaluation of merged checkpoints on live RoboCasa tasks

It is not the RLDS / TFDS / LIBERO reproduction path.

## Main Entry Points

Training:

- `vla-scripts/finetune.py`

Offline merge:

- `vla-scripts/merge_lora_weights_and_save.py`

Evaluation:

- `experiments/robot/robocasa/run_robocasa_eval.py`

## RoboCasa Training Contract

The RoboCasa path uses a native LeRobot-backed dataset loader rather than RLDS.

Current conventions:

- primary image:
  - `observation.images.robot0_agentview_left`
- optional wrist image:
  - `observation.images.robot0_eye_in_hand`
- language:
  - `annotation.human.task_description`
- action:
  - 7D OpenVLA-OFT contract
  - `[:3]` end-effector position
  - `[3:6]` end-effector rotation
  - `[6]` gripper
- optional proprio:
  - built from RoboCasa relative end-effector state and gripper state

The same contract is used by the RoboCasa evaluator.

## Checkpoint Format

Routine RoboCasa checkpoints are now adapter-only by default.

This is intentional:

- faster checkpoint saves
- safer behavior under Slurm walltime/preemption
- direct resumability without repeated full-model merges

A training checkpoint contains:

- `lora_adapter/`
- `action_head--latest_checkpoint.pt`
- optional `proprio_projector--latest_checkpoint.pt`
- `trainer_state--latest_checkpoint.pt`
- `dataset_statistics.json`
- `base_model_path.txt`

It does not contain merged `model-*.safetensors` unless you explicitly set:

- `--merge_lora_during_training True`

## Latest Checkpoint Semantics

`latest_chkpt` is now a symlink to the newest fully completed numbered checkpoint.

That means:

- saves happen into a temp directory first
- the completed checkpoint is promoted atomically
- `latest_chkpt` is updated only after the new checkpoint is valid
- the previous numbered checkpoint is deleted only after the new one is promoted

So routine checkpointing is Slurm-safer than the old overwrite-in-place scheme.

## Fine-Tuning Command

Example single-task RoboCasa fine-tuning command:

```bash
cd ~/scratch/VLA_RL/openvla-oft

conda activate opvla_rbcasa

torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_backend lerobot \
  --data_root_dir ~/scratch/VLA_RL/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
  --dataset_name robocasa_pickplace_counter_to_cabinet \
  --run_root_dir runs/robocasa \
  --use_l1_regression True \
  --use_diffusion False \
  --use_proprio False \
  --num_images_in_input 1 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --max_steps 10000 \
  --save_freq 1000 \
  --run_id_note robocasa_pickplace
```

## Resume Behavior

Resume works directly from an adapter-only checkpoint:

- load base model from `base_model_path.txt`
- load `lora_adapter/`
- restore `action_head` / optional `proprio_projector`
- restore optimizer, scheduler, and step from `trainer_state--latest_checkpoint.pt`

CLI parameters still control the new invocation, including:

- batch size
- max steps for this invocation
- learning rate

## Offline Merge for Evaluation

Before evaluation, merge the adapter into the base model offline.

The repo already provides:

- `vla-scripts/merge_lora_weights_and_save.py`

For new checkpoints, the script can infer the original base model from:

- `base_model_path.txt`

Example:

```bash
cd ~/scratch/VLA_RL/openvla-oft

conda activate opvla_rbcasa

python vla-scripts/merge_lora_weights_and_save.py \
  --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT_DIR
```

Optional explicit override:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT_DIR
```

After this step, the checkpoint directory contains merged `model-*.safetensors` and is ready for evaluation.

## Evaluation Command

Example RoboCasa evaluation command:

```bash
cd ~/scratch/VLA_RL/openvla-oft

conda activate opvla_rbcasa

python \
  -m experiments.robot.robocasa.run_robocasa_eval \
  --pretrained_checkpoint /PATH/TO/MERGED_CHECKPOINT_DIR \
  --dataset_name robocasa_pickplace_counter_to_cabinet \
  --task_name PickPlaceCounterToCabinet \
  --split target \
  --num_trials 1 \
  --max_steps 100 \
  --use_wandb False \
  --save_rollout_videos True \
  --center_crop True \
  --use_proprio False \
  --num_images_in_input 1 \
  --num_open_loop_steps 1
```

If you want to be explicit, also pass:

```bash
--unnorm_key robocasa_pickplace_counter_to_cabinet
```

## Expected Workflow

The intended end-to-end flow is:

1. set up the environment and assets via [ROBOCASA_SETUP.md](./ROBOCASA_SETUP.md)
2. fine-tune on RoboCasa with `vla-scripts/finetune.py`
3. resume from adapter-only checkpoints as needed
4. merge a chosen checkpoint offline with `vla-scripts/merge_lora_weights_and_save.py`
5. evaluate the merged checkpoint with `run_robocasa_eval.py`

