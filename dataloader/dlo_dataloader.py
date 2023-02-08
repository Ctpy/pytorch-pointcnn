import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from dataset.dlo_dataset import DLODataset


class DLOModule(pl.LightningDataModule):
    """
    DLO dataloader
    """

    def __init__(self, data_dir: str = "./", batch_size=2):
        super().__init__()
        self.dlo_predict = None
        self.dlo_test = None
        self.dlo_val = None
        self.dlo_train = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([])

    def prepare_data(self) -> None:
        # Download dataloader
        pass

    def setup(self, stage: str) -> None:

        if stage == "fit":
            dlo_full = DLODataset(self.data_dir, train=True)
            self.dlo_train = np.asarray(dlo_full)[:1]
            self.dlo_val = np.asarray(dlo_full)[1:5] #random_split(dlo_full, [1, len(dlo_full) - 1])
            
        if stage == "test":
            self.dlo_test = DLODataset(self.data_dir, train=False)
        
        if stage == "predict":
            self.dlo_predict = DLODataset(self.data_dir, train=False)
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dlo_train, self.batch_size, num_workers=2)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dlo_val, self.batch_size, num_workers=2)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dlo_test, self.batch_size, num_workers=2)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dlo_predict, self.batch_size, num_workers=2)
    
    