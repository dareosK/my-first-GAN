import os
import json
import hydra
import importlib
import torch
import torch.nn as nn
import neptune.new as neptune


from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

from lightning.source.modules import *
from lightning.source.tasks import *
from lightning.source.datamodules import *
from lightning.source.modules import Generator, Discriminator
from lightning.source.logging import FoolProofNeptuneLogger
from lightning.source.utils.general import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# globals:
import pandas as pd
pd.options.mode.use_inf_as_na = True
import pytorch_lightning as pl
pl.seed_everything(69)  # the number of doom


assert os.environ['NEPTUNE_API_TOKEN'], 'No Neptune API Token found. Please export your api token with `export NEPTUNE_API_TOKEN=<token>`.'
config_path = "/home/ruyogagp/gan/waifu/lightning/source/config"

def train(FLAGS):

    DataModule = eval(FLAGS.experiment.datamodule)

    ### initialize datamodule
    datamodule = DataModule(transforms=None,
                            **FLAGS.experiment.datamodule_kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    gen = Generator(block_activation=nn.ReLU,
                    final_activation=nn.Tanh,
                    **FLAGS.experiment.generator_kwargs)
    disc = Discriminator(block_activation=nn.LeakyReLU,
                         final_activation=nn.Sigmoid,
                         **FLAGS.experiment.discriminator_kwargs)
    # initialize Task
    task = GAN(generator=gen,
                discriminator=disc,
                optimizer_kwargs=dict(lr=1e-4,beta1=0.5,beta2=0.9999, weight_decay=1e-3))

    # initialize trainer
    callbacks = get_default_callbacks(monitor=FLAGS.experiment.monitor)

    trainer = pl.Trainer(**FLAGS.trainer,
                         callbacks=callbacks,
                         logger=set_up_neptune(FLAGS))

    # run
    trainer.fit(task, datamodule)


@hydra.main(config_path, config_name="gan")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    FLAGS.config_path = config_path
    return train(FLAGS)

if __name__ == '__main__':
    main()


