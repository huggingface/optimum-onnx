import warnings

from optimum.onnxruntime.modeling import *  # noqa F403


warnings.warn(
    "Importing from `optimum.onnxruntime.modeling_ort` is deprecated and will be removed in the next major release. "
    "Please import from `optimum.onnxruntime.modeling` instead.",
    FutureWarning,
    stacklevel=2,
)
