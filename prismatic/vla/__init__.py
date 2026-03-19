"""VLA package exports.

Keep dataset / RLDS imports lazy so inference-only code can import lightweight
submodules such as `action_tokenizer` and `constants` without pulling in the
TensorFlow-backed training stack.
"""


def get_vla_dataset_and_collator(*args, **kwargs):
    from .materialize import get_vla_dataset_and_collator as _get_vla_dataset_and_collator

    return _get_vla_dataset_and_collator(*args, **kwargs)
