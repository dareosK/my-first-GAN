import pytorch_lightning as pl

import torchvision.datasets as TorchvisionDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.utils as TorchVisionUtils


class WaifuDataModule(pl.LightningDataModule):
    def __init__(self, folderpath, batch_size=128, image_size=64, transforms=None):
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
                           transforms.Compose([transforms.Resize(image_size),
                           transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dataset = TorchvisionDataset.ImageFolder(root=folderpath,
                                                      transform=self.transforms)
        self.data = None

    def setup(self, stage=None):
        self.data = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=4)

    def preview_batch(self):
        """
        Preview a batch of waifus
        :return: None
        """
        if self.data is None:
            self.setup()
        # Get batch
        batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(TorchVisionUtils.make_grid(batch[0].to('cpu')[:64],
                                                           padding=2,
                                                           normalize=True).cpu(), (1, 2, 0)))


