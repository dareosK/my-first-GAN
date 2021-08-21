import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch

from lightning.source.datamodules import *
from lightning.source.modules import *

class GAN(LightningModule):
    def __init__(self, discriminator, generator, batch_size, optimizer_kwargs, **kwargs):
        """
        :param discriminator: discriminator network
        :param generator: generator network
        :param batch_size: batch size for the training
        :param optimizer_kwargs: `dict` keyword arguments for the optimizer
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.batch_size = batch_size
        self.optimizer_kwargs = optimizer_kwargs

    def configure_optimizers(self):
        lr = self.hparams.optimizer_kwargs['lr']
        beta1 = self.hparams.optimizer_kwargs['beta1']
        beta2 = self.hparams.optimizer_kwargs['beta2']

        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        return [opt_generator, opt_discriminator], []

    def loss(self, output, label):
        return F.binary_cross_entropy(output, label)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch

        # sample noise
        z = torch.randn(images.shape[0], self.latent_dim)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_images = self(z)

            # log sampled images
            sample_images = self.generated_images[:6]
            grid = torchvision.utils.make_grid(sample_images)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake_label)
            true_label = torch.ones(images.size(0), 1)

            # adversarial loss is binary cross-entropy
            g_loss = self.loss(self.discriminator(self(z)), true_label)
            self.log("generator_loss", g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            output = dict(loss=g_loss)
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            true_label = torch.ones(images.size(0), 1)

            real_loss = self.loss(self.discriminator(images), true_label)

            # Make fake_label label with size (batchsize, 1)
            fake_label = torch.zeros(images.size(0), 1)

            fake_loss = self.loss(
                self.discriminator(self(z).detach()), fake_label)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("generator_loss", d_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            output = dict(loss=d_loss)
            return output


    def forward(self, z):
        return self.generator(z)
