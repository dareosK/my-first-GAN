import pytorch_lightning as pl

import torchvision.datasets as TorchvisionDataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.utils as TorchVisionUtils


class WaifuDatamodule(pl.LightningDataModule):
    def __init__(self, folderpath='/Users/gregruyoga/G_PROJECTS/img_database/waifu', batch_size=128, image_size=64, transforms=None):
        """"
        :parameter folderpath: folder containing waifu images
        :parameter image_size: image size to centercrop + resize to
        :parameter transforms: transormations to apply to tha images
        """
        super().__init__(self)
        self.folderpath = folderpath
        self.batch_size = batch_size
        self.image_size = image_size
        self.transforms =  transforms if transforms is not None else \
                           T.Compose([T.Resize(image_size),
                           T.CenterCrop(image_size),
                           T.ToTensor(),
                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataloader = None

    def setup(self, stage=None):
        self.dataset = TorchvisionDataset.ImageFolder(root=self.folderpath,
                                                      transform=self.transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=4)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def get_batch(self):
        if self.dataloader is None:
            self.setup()
        # The syntax is weird but this is how you get a batch from dataloader
        batch = next(iter(self.dataloader))
        return batch

    def preview_batch(self, normalize=True):
        """
        Preview a batch of waifus
        """
        batch = self.get_batch()
#         creates a grid of 8*8
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(TorchVisionUtils.make_grid(batch[0].to('cpu')[:64],
                                                           padding=2,
                                                           normalize=normalize).cpu(), (1, 2, 0)))