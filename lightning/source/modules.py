import pytorch_lightning as pl
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, featuremap_dim, output_dim=3, activation=None, final_activation=None):
        """
        Standard Generator
        :parameter latent_dim: `int` latent dimension of the generator
        :parameter featuremap_dim: `int` dimension of the feature map, starts at featuremap_dim * 8 on the first layer
        :parameter output_dim: `int` output dimension of the generator
        :parameter activation: `torch.nn` activation function for the intermediate layers
        :parameter final_activation: `torch.nn` activation of the final layer

        """
        self.latent_dim = latent_dim
        self.featuremap_dim = featuremap_dim
        self.output_dim = output_dim
        self.activation = activation
        self.final_activation = final_activation

        if activation is None:
            self.activation = nn.ReLU

        if final_activation is None:
            self.final_activation = nn.Sigmoid

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, featuremap_dim * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(featuremap_dim * 8),
            self.activation(inplace=True),

            nn.ConvTranspose2d(featuremap_dim * 8, featuremap_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(featuremap_dim * 4),
            self.activation(inplace=True),

            nn.ConvTranspose2d(featuremap_dim * 4, featuremap_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(featuremap_dim * 2),
            self.activation(inplace=True),

            nn.ConvTranspose2d(featuremap_dim * 2, featuremap_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(featuremap_dim),
            self.activation(inplace=True),

            nn.ConvTranspose2d(featuremap_dim * 2, featuremap_dim, kernel_size=4, stride=2, padding=1),
            self.final_activation
        )