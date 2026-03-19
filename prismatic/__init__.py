"""Top-level prismatic exports.

Use lazy wrappers so importing `prismatic` does not eagerly import optional
training/data dependencies.
"""


def available_models():
    from .models import available_models as _available_models

    return _available_models()


def available_model_names():
    from .models import available_model_names as _available_model_names

    return _available_model_names()


def get_model_description(*args, **kwargs):
    from .models import get_model_description as _get_model_description

    return _get_model_description(*args, **kwargs)


def load(*args, **kwargs):
    from .models import load as _load

    return _load(*args, **kwargs)
