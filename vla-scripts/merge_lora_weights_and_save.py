"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base OpenVLA model. Saves the final checkpoint in the same directory.

If the checkpoint was produced by the latest fine-tuning pipeline, this script can infer
the original base model from `base_model_path.txt` in the checkpoint directory. You can
still override it explicitly with `--base_checkpoint` if needed.

Make sure to specify the correct base checkpoint when overriding. For example,
- if you fine-tuned the default OpenVLA-7B model without modifications, then `--base_checkpoint=="openvla/openvla-7b"`
- if you fine-tuned a different model or resumed fine-tuning from a different checkpoint, then specify that base checkpoint
- if you fine-tuned the default OpenVLA-7B model with modifications to `modeling_prismatic.py` (OpenVLA class definition),
  then the base checkpoint path should point to the checkpoint containing the modifications

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/

    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel

# Keep this script on the PyTorch-only Transformers path. The RoboCasa env may
# have an intentionally broken TensorFlow stack because training/eval here do
# not depend on TFDS/RLDS.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoConfig, AutoModelForVision2Seq

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from utilities import load_base_model_path_from_checkpoint


@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Optional base model checkpoint override
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter

    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    checkpoint_dir = Path(cfg.lora_finetuned_checkpoint_dir)
    assert checkpoint_dir.is_dir(), f"Checkpoint directory does not exist: {checkpoint_dir}"

    base_checkpoint = str(cfg.base_checkpoint).strip()
    if not base_checkpoint:
        inferred_base_checkpoint = load_base_model_path_from_checkpoint(checkpoint_dir)
        assert (
            inferred_base_checkpoint is not None
        ), "Could not infer base checkpoint from base_model_path.txt; please provide --base_checkpoint explicitly."
        base_checkpoint = inferred_base_checkpoint

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load Model using HF AutoClasses
    print(f"Loading base model: {base_checkpoint}")
    vla = AutoModelForVision2Seq.from_pretrained(
        base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, checkpoint_dir / "lora_adapter").to("cuda")
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(checkpoint_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{checkpoint_dir}")


if __name__ == "__main__":
    main()
