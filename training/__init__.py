"""Training pipeline: loop, loss, scheduler, checkpoints.

``Trainer`` owns the training loop; ``lm_cross_entropy`` the loss;
``build_lr_scheduler`` the warmup+cosine schedule; ``save_checkpoint`` /
``load_checkpoint`` persistence. Entry point CLI lives in
:mod:`training.train`.
"""

from training.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from training.loss import lm_cross_entropy
from training.scheduler import build_lr_scheduler, lr_prefactor
from training.trainer import Trainer

__all__ = [
    "Checkpoint",
    "Trainer",
    "build_lr_scheduler",
    "lm_cross_entropy",
    "load_checkpoint",
    "lr_prefactor",
    "save_checkpoint",
]
