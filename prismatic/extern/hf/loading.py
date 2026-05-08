"""Shared Hugging Face loading helpers for OpenVLA and MiniVLA-compatible checkpoints."""

from __future__ import annotations

from typing import Any, Optional

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoConfig, AutoModelForVision2Seq, AutoTokenizer

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, PrismaticConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, PrismaticForConditionalGeneration
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer


def register_prismatic_auto_classes() -> None:
    """Register both OpenVLA and base Prismatic model classes with HF Auto classes."""
    registrations = [
        (AutoConfig.register, ("prismatic", PrismaticConfig)),
        (AutoConfig.register, ("openvla", OpenVLAConfig)),
        (AutoModelForVision2Seq.register, (PrismaticConfig, PrismaticForConditionalGeneration)),
        (AutoModelForVision2Seq.register, (OpenVLAConfig, OpenVLAForActionPrediction)),
    ]
    for register_fn, args in registrations:
        try:
            register_fn(*args)
        except ValueError:
            continue


def infer_openvla_config(processor: Any, config: Any) -> OpenVLAConfig:
    """Convert a base Prismatic config into an OpenVLA config with action metadata."""
    tokenizer = processor.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer)

    if isinstance(config, OpenVLAConfig):
        config.action_token_begin_idx = action_tokenizer.action_token_begin_idx
        config.stop_token_id = action_tokenizer.stop_token_id
        config.prompt_suffix_token_id = action_tokenizer.prompt_suffix_token_id
        return config

    config_dict = config.to_dict()
    config_dict["norm_stats"] = config_dict.get("norm_stats")
    config_dict["n_action_bins"] = config_dict.get("n_action_bins", action_tokenizer.vocab_size)
    config_dict["action_token_begin_idx"] = action_tokenizer.action_token_begin_idx
    config_dict["stop_token_id"] = action_tokenizer.stop_token_id
    config_dict["prompt_suffix_token_id"] = action_tokenizer.prompt_suffix_token_id

    return OpenVLAConfig(**config_dict)


def load_prismatic_processor(
    pretrained_checkpoint: str,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
):
    register_prismatic_auto_classes()
    image_processor = PrismaticImageProcessor.from_pretrained(
        pretrained_checkpoint,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_checkpoint,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_action_prediction_model(
    pretrained_checkpoint: str,
    *,
    processor_path: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    **from_pretrained_kwargs: Any,
):
    """Load a checkpoint as an action-prediction model, coercing base Prismatic checkpoints when needed."""
    register_prismatic_auto_classes()

    processor = load_prismatic_processor(
        processor_path or pretrained_checkpoint,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        pretrained_checkpoint,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )

    openvla_config = infer_openvla_config(processor, config)
    model = OpenVLAForActionPrediction.from_pretrained(
        pretrained_checkpoint,
        config=openvla_config,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    return model, processor
