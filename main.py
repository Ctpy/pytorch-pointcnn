import argparse
import torch
import wandb
import numpy as np
from model.pointcnn_seg import PointCNNSeg
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataloader.wireharness_dataloader import WireHarnessModule
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PointCNN',
        description='Set parameters for training')
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    data_path = args.data
    num_classes = args.num_classes
    split = args.split
    batch_size = args.batch_size

    # Loading the data
    dl = WireHarnessModule(data_path, batch_size=batch_size)
    dl.setup(split)
    train_set = dl.train_dataloader()
    val_set = dl.val_dataloader()
    print("Train batches:", len(train_set))
    print("Val batches:", len(val_set))

    # Start training
    wandb.login()
    torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(dirpath="output/ckpt", save_top_k=5, monitor="val_loss_epoch")
    # early_stop_callback = EarlyStopping(monitor="val_loss_epoch", patience=10, mode="min")
    model = PointCNNSeg(num_classes=num_classes, weight_balance=dl.get_class_balance_weights(num_classes=num_classes))
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=200, accelerator='gpu', devices=1, logger=wandb_logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=val_set)

