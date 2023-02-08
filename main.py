from dataloader.dlo_dataloader import DLOModule
from model.pointcnn import PointCNN
import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    dl = DLOModule("C:\dataset")
    dl.setup("fit")
    train_set = dl.train_dataloader()
    val_set = dl.val_dataloader()
    print("Train batches:", len(train_set))
    print("Val batches:", len(val_set))
    dl.setup("test")
    # test_set = dl.test_dataloader()
    # print("Test batches:", len(test_set))

    model = PointCNN()
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1)
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)