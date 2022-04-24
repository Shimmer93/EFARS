# Credit: https://github.com/facebookresearch/TimeSformer/tree/main/timesformer/models

from .build import MODEL_REGISTRY, build_model  # noqa
from .custom_video_model_builder import *  # noqa
from .video_model_builder import ResNet, SlowFast # noqa
