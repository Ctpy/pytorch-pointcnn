import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from dataset.wireharness_dataset import WireHarenessDataset


class WireHarnessModule(pl.LightningDataModule):
    """
    wire Harness dataloader
    """

    def __init__(self, data_dir: str = "./", batch_size=2):
        super().__init__()
        self.predict = None
        self.test = None
        self.val = None
        self.train = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([])

    def prepare_data(self) -> None:
        # Download dataloader
        pass

    def setup(self, stage: str) -> None:

        if stage == "train":
            self.train = WireHarenessDataset(self.data_dir, split="train", transform=True)
            self.val = WireHarenessDataset(self.data_dir, split="val", transform=False)
            
        if stage == "test":
            self.test = WireHarenessDataset(self.data_dir, split="test")
        
        if stage == "predict":
            self.predict = WireHarenessDataset(self.data_dir, split="test")

    def get_class_balance_weights(self, num_classes=3):
        label_weights = np.zeros(num_classes)
        for wh_file in self.train:
            label = wh_file['label']
            weights, _ = np.histogram(label, range=(0, 5))
            label_weights += np.array([weights[0] + weights[1] + weights[3], weights[2], weights[4]])
        label_weights = np.log(label_weights / np.sum(label_weights))
        label_weights = label_weights / np.sum(label_weights)
        label_weights[-1] = 0
        return label_weights
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, self.batch_size, num_workers=2)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, self.batch_size, num_workers=2)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, self.batch_size, num_workers=2)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict, self.batch_size, num_workers=2)
    
    