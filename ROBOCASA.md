# RoboCasa Evaluation

This document describes the thin RoboCasa integration added to `openvla-oft`.

The goal of this integration is narrow:
- run a pretrained OpenVLA-OFT checkpoint on a single RoboCasa task
- keep RoboCasa data and environment usage independent of TFDS / RLDS
- avoid building a full RoboCasa training pipeline inside `openvla-oft`

## What Was Added

The main entry point is:

- `experiments/robot/robocasa/run_robocasa_eval.py`

This script mirrors the structure of the existing LIBERO evaluator, but swaps in a RoboCasa environment adapter.

## Integration Design

The integration works as follows:

1. A RoboCasa environment is created through the RoboCasa Gym wrapper.
2. RoboCasa observations are converted into the OpenVLA-OFT eval input format.
3. The model predicts an action chunk.
4. The predicted 7D actions are mapped into RoboCasa action fields.
5. Rollouts are logged and can optionally be saved as MP4 videos.

### Observation Mapping

The evaluator currently uses:

- `video.robot0_agentview_left` as the main image
- `video.robot0_eye_in_hand` as the wrist image
- `annotation.human.task_description` as the language instruction

If `use_proprio=True`, the evaluator also builds a proprio vector from:

- `state.end_effector_position_relative`
- `state.end_effector_rotation_relative`
- `state.gripper_qpos`

### Action Mapping

The evaluator assumes OpenVLA-OFT emits 7D actions:

- `[:3]` end-effector position delta
- `[3:6]` end-effector rotation delta
- `[6]` gripper action

These are mapped to the RoboCasa Gym action dict:

- `action.end_effector_position`
- `action.end_effector_rotation`
- `action.gripper_close`
- `action.base_motion`
- `action.control_mode`

Base motion is fixed to zeros in the current thin integration.

## Why `openvla_utils.py` Was Changed

The RoboCasa eval flow needed `openvla-oft` to behave well in an inference-only setting.

The relevant changes were:

- lazy `prismatic` imports so inference does not eagerly pull in the RLDS / TFDS training stack
- PIL-based image resize / crop in the eval path instead of TensorFlow image ops
- better handling of cached Hugging Face checkpoints in offline / local-cache scenarios
- CPU-safe loading of component checkpoints when CUDA is unavailable

These changes were made to support the RoboCasa evaluation path and to keep TFDS / RLDS out of this inference flow.

## Environment Prerequisites

Before running RoboCasa evaluation, RoboCasa setup must be completed:

```bash
conda activate opvla_rbcasa
cd /storage/ice1/4/5/asinha389/VLA_RL/robocasa

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

In addition, `robosuite` macros should exist as well:

```bash
cd /storage/ice1/4/5/asinha389/VLA_RL/robosuite
python -m robosuite.scripts.setup_macros
```

## Running RoboCasa Evaluation

The pretrained checkpoint that was validated during integration was:

- `moojink/openvla-7b-oft-finetuned-libero-10`

The important `unnorm_key` for that checkpoint is:

- `libero_10_no_noops`

### Short Test Rollout

```bash
cd /storage/ice1/4/5/asinha389/VLA_RL/openvla-oft

/storage/ice1/4/5/asinha389/.conda/envs/opvla_rbcasa/bin/python \
  -m experiments.robot.robocasa.run_robocasa_eval \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --unnorm_key libero_10_no_noops \
  --task_name PickPlaceCounterToCabinet \
  --split target \
  --num_trials 1 \
  --max_steps 5 \
  --use_wandb False \
  --save_rollout_videos False \
  --center_crop True \
  --use_proprio False \
  --num_images_in_input 1 \
  --num_open_loop_steps 1
```

### Longer Rollout With Saved Video

```bash
cd /storage/ice1/4/5/asinha389/VLA_RL/openvla-oft

/storage/ice1/4/5/asinha389/.conda/envs/opvla_rbcasa/bin/python \
  -m experiments.robot.robocasa.run_robocasa_eval \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --unnorm_key libero_10_no_noops \
  --task_name PickPlaceCounterToCabinet \
  --split target \
  --num_trials 1 \
  --max_steps 100 \
  --use_wandb False \
  --save_rollout_videos True \
  --center_crop True \
  --use_proprio False \
  --num_images_in_input 1 \
  --num_open_loop_steps 1 \
  --run_id_note long_rollout_video
```

## Running RoboCasa Fine-Tuning

A native LeRobot-backed RoboCasa training path is now available through the main `vla-scripts/finetune.py` entrypoint.

Use the new dataset backend flag:

- `data_backend=lerobot`

For the current thin training integration, `data_root_dir` should point directly to a RoboCasa LeRobot dataset root, for example:

- `/storage/ice1/4/5/asinha389/VLA_RL/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot`

The training path currently:

- reads RoboCasa LeRobot `meta/`, `data/*.parquet`, and `videos/*.mp4` directly
- uses `robot0_agentview_left` as the primary image
- uses `robot0_eye_in_hand` as the optional wrist image
- maps RoboCasa actions to the 7D OpenVLA-OFT arm action expected by the evaluator
- saves an OpenVLA-compatible `dataset_statistics.json` so the resulting checkpoint can be evaluated with a RoboCasa-specific `unnorm_key`

### Example Single-Task RoboCasa Fine-Tuning Command

Use `torchrun`, matching the upstream `openvla-oft` training flow:

```bash
cd /storage/ice1/4/5/asinha389/VLA_RL/openvla-oft

/storage/ice1/4/5/asinha389/.conda/envs/opvla_rbcasa/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_backend lerobot \
  --data_root_dir /storage/ice1/4/5/asinha389/VLA_RL/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
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
  --wandb_project openvla-oft-robocasa \
  --run_id_note robocasa_pickplace
```

### Validated Smoke Test Command

This shorter command was validated locally on the RoboCasa LeRobot dataset with per-iteration timing enabled:

```bash
cd /storage/ice1/4/5/asinha389/VLA_RL/openvla-oft

WANDB_MODE=disabled /storage/ice1/4/5/asinha389/.conda/envs/opvla_rbcasa/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-10 \
  --data_backend lerobot \
  --data_root_dir /storage/ice1/4/5/asinha389/VLA_RL/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
  --dataset_name robocasa_pickplace_counter_to_cabinet \
  --run_root_dir runs/robocasa_smoke \
  --use_l1_regression True \
  --use_diffusion False \
  --use_proprio False \
  --num_images_in_input 1 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --max_steps 3 \
  --save_freq 100 \
  --wandb_project openvla-oft-robocasa \
  --run_id_note smoke_timed \
  --log_step_timing True
```

### Evaluating a RoboCasa-Finetuned Checkpoint

If the checkpoint was trained with the example command above, the RoboCasa evaluator should use:

- `--unnorm_key robocasa_pickplace_counter_to_cabinet`

This works because the LeRobot training backend saves `dataset_statistics.json` with that dataset key.

## Current Status

The integration is working at the infrastructure level:

- the model loads
- RoboCasa env creation works
- observations are converted
- actions are produced and stepped in the environment
- rollout videos can be saved

The current limitation is behavioral:

- the LIBERO-finetuned checkpoint does not solve the RoboCasa task out of the box
- action semantics and scaling likely need additional tuning for RoboCasa

## Main Tuning Knobs

If you want to improve behavior, the current evaluator exposes:

- `action_scale_pos`
- `action_scale_rot`
- `action_clip_pos`
- `action_clip_rot`
- `control_mode`
- `num_open_loop_steps`
- `use_proprio`

These are the most likely places to adjust the thin integration before considering any RoboCasa-specific fine-tuning.
