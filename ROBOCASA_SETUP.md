# RoboCasa Setup

This document explains how to create a fresh environment for RoboCasa training and evaluation with `openvla-oft`.

Use this setup if your goal is:

- fine-tuning OpenVLA-OFT on RoboCasa data
- resuming RoboCasa fine-tuning
- exporting merged checkpoints for RoboCasa evaluation
- evaluating merged checkpoints on live RoboCasa tasks

Do not use this environment for exact upstream RLDS / TFDS LIBERO reproduction. That should live in a separate env.

## Workspace Layout

Clone the repos as siblings in the same workspace directory.

Example:

```bash
mkdir -p ~/scratch/VLA_RL
cd ~/scratch/VLA_RL

git clone <your-openvla-oft-remote> openvla-oft
git clone https://github.com/robocasa/robocasa.git robocasa
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
```

The expected layout is:

```text
VLA_RL/
  openvla-oft/
  robocasa/
  robosuite/
```

This matters because the RoboCasa evaluator expects `robocasa` and `robosuite` to exist parallel to `openvla-oft`.

## Why This Needs a Special Environment

`openvla-oft` and `robocasa` have conflicting upstream dependency stacks.

`openvla-oft` upstream expects older pinned packages such as:

- `torch==2.2.0`
- `tensorflow==2.15.0`
- `tensorflow_datasets==4.9.3`
- `draccus==0.8.0`

`robocasa` expects a newer simulation / LeRobot stack such as:

- `numpy==2.2.5`
- `mujoco==3.3.1`
- `lerobot==0.3.3`

If you install both repos with all upstream dependencies enabled, the environment drifts and the TensorFlow / RLDS stack breaks.

For RoboCasa train/eval, that is acceptable, because the RoboCasa path in this repo has been patched to avoid TFDS / RLDS imports.

The practical rule is:

- install `robosuite` normally
- install `robocasa` normally
- install `openvla-oft` in editable mode without dependencies
- then install only the `openvla-oft` runtime packages needed for RoboCasa

## Fresh Conda Environment

Recommended env name:

- `opvla_rbcasa`

Create it with:

```bash
conda create -n opvla_rbcasa python=3.10 -y
conda activate opvla_rbcasa
```

## Install Order

Install `robosuite` first:

```bash
cd ~/scratch/VLA_RL/robosuite
pip install -e .
```

Install `robocasa` next:

```bash
cd ~/scratch/VLA_RL/robocasa
pip install -e .
```

Install `openvla-oft` without its pinned dependencies:

```bash
cd ~/scratch/VLA_RL/openvla-oft
pip install -e . --no-deps
```

Then install the runtime packages needed for RoboCasa training and evaluation:

```bash
pip install \
  "accelerate>=0.25.0" \
  "draccus==0.10.0" \
  "einops" \
  "huggingface_hub" \
  "json-numpy" \
  "jsonlines" \
  "matplotlib" \
  "peft==0.11.1" \
  "protobuf" \
  "rich" \
  "sentencepiece==0.1.99" \
  "timm==0.9.10" \
  "tokenizers==0.19.1" \
  "wandb" \
  "diffusers==0.30.3" \
  "imageio" \
  "uvicorn" \
  "fastapi"
```

Install the OpenVLA-OFT Transformers fork:

```bash
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
```

Important:

- do not install the upstream `openvla-oft` TensorFlow / TFDS / RLDS dependencies in this environment
- do not use this environment for exact upstream LIBERO RLDS training reproduction

## RoboCasa Assets and Macros

Before using RoboCasa, set up macros and assets.

RoboCasa:

```bash
conda activate opvla_rbcasa
cd ~/scratch/VLA_RL/robocasa

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

robosuite:

```bash
cd ~/scratch/VLA_RL/robosuite
python -m robosuite.scripts.setup_macros
```

This creates the required private macro files and downloads the kitchen asset pack used by RoboCasa.

## Optional Visualization / LIBERO Notes

If you also want LIBERO visualization or notebook support in the same environment, you may need LIBERO-compatible versions of some packages, especially `robosuite`.

That is separate from RoboCasa fine-tuning itself. If your goal is only RoboCasa train/eval, keep this environment minimal.

## Sanity Checks

From `~/scratch/VLA_RL/openvla-oft`, these checks should work:

```bash
conda activate opvla_rbcasa

python -c "import robocasa; print('robocasa ok')"
python -c "from prismatic.vla.datasets import LeRobotDataset; print('lerobot backend ok')"
python -c "from experiments.robot.robocasa import run_robocasa_eval; print('robocasa eval import ok')"
```

If those imports succeed, the environment is ready for RoboCasa training and evaluation.

## Next Step

After setup, continue with the workflow guide:

- [ROBOCASA.md](./ROBOCASA.md)
