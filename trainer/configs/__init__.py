from .train_config import _C as conf_train
from .inference_config import _C as conf_val

from .train_config_sequence_06 import _C as conf_train_sync06
from .train_config_ddp_debug import _C as conf_train_sync06_debug


__all__ = ["cfg"]