import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
from neptune.new.types import File

from lightning.source.datamodules import *
from lightning.source.modules import *

class GAN(LightningModule):
    def __init__(self, discriminator, generator, optimizer_kwargs, **kwargs):
        """
        :param discriminator: discriminator network
        :param generator: generator network
        :param optimizer_kwargs: `dict` keyword arguments for the optimizer
        """
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.optimizer_kwargs = optimizer_kwargs
        self.fixed_noise = torch.randn(128, self.generator.latent_dim, 1, 1, device=self.device)
        self.save_hyperparameters()

    def configure_optimizers(self):
        lr = self.optimizer_kwargs['lr']
        beta1 = self.optimizer_kwargs['beta1']
        beta2 = self.hparams.optimizer_kwargs['beta2']

        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        return [opt_generator, opt_discriminator], []

    def loss(self, output, label):
        return F.binary_cross_entropy(output, label)

    def training_epoch_end(self, training_step_outputs):
        sample_image = np.transpose(TorchVisionUtils.make_grid(self.generated_images.to('cpu')[:64],
                                                               padding=2,
                                                               normalize=True).cpu(), (1, 2, 0))

        self.logger.experiment[f'images/gen-epoch{self.current_epoch}'].upload(File.as_image(sample_image))


    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch[0]

        # sample noise with shape (batchsize, latent_dim)
        z = torch.randn(images.shape[0], self.generator.latent_dim, 1, 1, device=self.device)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_images = self.forward(z)

            if self.global_step % 50 == 0:
                fixed_image = self.generator(self.fixed_noise.to(self.device))
                fixed_image = np.transpose(TorchVisionUtils.make_grid(fixed_image[0].to('cpu')[:64],
                                                                      padding=2,
                                                                      normalize=True).cpu(), (1, 2, 0))
                self.logger.experiment[f'images/fixed-step{self.global_step}'].upload(File.as_image(fixed_image))

            # ground truth result (ie: all fake_label)
            true_label = torch.ones(images.size(0), device=self.device)

            # adversarial loss is binary cross-entropy
            g_loss = self.loss(self.discriminator(self(z)).view(-1), true_label)
            self.log("generator_loss", g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            output = dict(loss=g_loss)
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            true_label = torch.ones(images.size(0), device=self.device)

            real_loss = self.loss(self.discriminator(images).view(-1), true_label)

            # Make fake_dlabel label with size (batchsize, 1)
            fake_label = torch.zeros(images.size(0), device=self.device)

            fake_loss = self.loss(
                self.discriminator(self.forward(z).detach()).view(-1), fake_label)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("discriminator_loss", d_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            output = dict(loss=d_loss)
            return output


    def forward(self, z):
        return self.generator(z)