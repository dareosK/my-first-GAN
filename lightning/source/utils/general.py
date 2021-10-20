import os
import json
import torch
import torch.nn as nn
import argparse
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, ListConfig, DictConfig

from lightning.source.callbacks import WriteCheckpointLogs
from lightning.source.logging import FoolProofNeptuneLogger



####################################################################################################
#                        neptune                                                                   #
####################################################################################################


def set_up_neptune(FLAGS={}, close_after_fit=False, **kwargs):
    """
    Set up a neptune logger from file.
    :param FLAGS: FLAGS namespace object
    :param close_after_fit:
    :param kwargs:
    :return:
    """
    if not "NEPTUNE_API_TOKEN" in os.environ:
        raise EnvironmentError('Please set environment variable `NEPTUNE_API_TOKEN`.')

    tags = FLAGS.neptune.tags if not isinstance(FLAGS.neptune.tags, (ListConfig,)) \
        else OmegaConf.to_container(FLAGS.neptune.tags, resolve=True)

    neptune_logger = FoolProofNeptuneLogger(api_key=os.environ["NEPTUNE_API_TOKEN"],
                                            close_after_fit=close_after_fit,
                                            project=FLAGS.neptune.project,
                                            tags=tags)
    return neptune_logger


def get_default_callbacks(monitor='generator_loss', monitor_mode='min', early_stop=False, early_stop_patience=25, **kwargs):
    """
    Instantate the default callbacks: EarlyStopping and Checkpointing (including logging of checkpoint path).

    :param monitor: monitor for checkpointing
    :param monitor_mode:
    :return:
    """
    checkpoint_path_callback = WriteCheckpointLogs()
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor=monitor, verbose=True,
                                                                        save_last=True, save_top_k=3,
                                                                        save_weights_only=False, mode=monitor_mode,
                                                                        every_n_val_epochs=1)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=False)
    if early_stop:
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                                               patience=early_stop_patience,
                                                               verbose=True, mode='min',
                                                               strict=False)
        return [checkpoint_path_callback, checkpoint_callback, early_stop, lr_monitor]
    else:
        return [checkpoint_path_callback, checkpoint_callback, lr_monitor]