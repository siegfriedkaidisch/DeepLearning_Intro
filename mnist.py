import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# tensorboard --logdir=logs --port=6007

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None):
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        print("Loaded images from disk.")
        print("Number of train,val,test images: ", train_size, val_size, test_size)
        print("────────────────────────────────────────────────────────────────────────────")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)


class MNISTModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr

        self.flatten = nn.Flatten()
        self.hl1 = nn.Linear(28 * 28, 128)
        self.act1 = nn.ReLU()
        self.hl2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hl1(x)
        x = self.act1(x)
        x = self.hl2(x)
        x = self.act2(x)
        x = self.output_layer(x)
        return x

    def training_step(self, batch):
        x, y = batch
        ann_output = self(x)
        loss = F.cross_entropy(ann_output, y)

        preds = torch.argmax(ann_output, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        ann_output = self(x)
        loss = F.cross_entropy(ann_output, y)

        preds = torch.argmax(ann_output, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def test_step(self, batch):
        x, y = batch
        ann_output = self(x)
        loss = F.cross_entropy(ann_output, y)

        preds = torch.argmax(ann_output, dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
        #return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    data_module = MNISTDataModule(data_dir="./images/")
    model = MNISTModel()

    print("────────────────────────────────────────────────────────────────────────────")
    trainer = Trainer(max_epochs=100, logger=TensorBoardLogger("logs/"))
    print("────────────────────────────────────────────────────────────────────────────")
    trainer.fit(model=model, datamodule=data_module)
    print("────────────────────────────────────────────────────────────────────────────")
    trainer.test(datamodule=data_module, ckpt_path='best')
