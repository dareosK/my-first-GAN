import pytorch_lightning as pl
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, featuremap_dim, output_dim=3, block_activation=None, final_activation=None, **kwargs):
        """
        Standard Generator
        :parameter latent_dim: `int` latent dimension of the generator
        :parameter featuremap_dim: `int` dimension of the feature map, starts at featuremap_dim * 8 on the first layer
        :parameter output_dim: `int` output dimension of the generator
        :parameter block_activation: `torch.nn` activation function for the intermediate layers
        :parameter final_activation: `torch.nn` activation of the final layer

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.featuremap_dim = featuremap_dim
        self.output_dim = output_dim

        self.block_activation = block_activation if block_activation is not None else nn.LeakyReLU
        self.final_activation = final_activation if final_activation is not None else nn.Tanh

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, featuremap_dim * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(featuremap_dim * 8),
            self.block_activation(),

            nn.ConvTranspose2d(featuremap_dim * 8, featuremap_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(featuremap_dim * 4),
            self.block_activation(),

            nn.ConvTranspose2d(featuremap_dim * 4, featuremap_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(featuremap_dim * 2),
            self.block_activation(),

            nn.ConvTranspose2d(featuremap_dim * 2, featuremap_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(featuremap_dim),
            self.block_activation(),

            nn.ConvTranspose2d(featuremap_dim, output_dim, kernel_size=4, stride=2, padding=1),
            self.final_activation()
        )

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class Discriminator(nn.Module):
    def __init__(self, featuremap_dim, input_dim=3, block_activation=None, final_activation=None, **kwargs):
        """
        Standard Discriminator
        :parameter featuremap_dim: `int` dimension of the feature map, starts at featuremap_dim * 8 on the first layer
        :parameter input_dim: `int` input dimension of the network
        :parameter activation: `torch.nn` activation function for the intermediate layers
        :parameter final_activation: `torch.nn` activation of the final layer
        """
        super().__init__()
        self.featuremap_dim = featuremap_dim
        self.input_dim = input_dim
        self.block_activation = block_activation if block_activation is not None else nn.LeakyReLU
        self.final_activation = final_activation if final_activation is not None else nn.Sigmoid

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, featuremap_dim, kernel_size=4, stride=2, padding=1, bias=False),
            self.block_activation(),

            nn.Conv2d(featuremap_dim, featuremap_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(featuremap_dim * 2),
            self.block_activation(),

            nn.Conv2d(featuremap_dim * 2, featuremap_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(featuremap_dim * 4),
            self.block_activation(),

            nn.Conv2d(featuremap_dim * 4, featuremap_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(featuremap_dim * 8),
            self.block_activation(),

            nn.Conv2d(featuremap_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            self.final_activation()
        )

    def forward(self, inputs):
        out = self.net(inputs)
        return out