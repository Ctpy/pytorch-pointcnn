import torch
import torchmetrics
from torch import nn
import torch_geometric.nn as tg
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class PointCNN(LightningModule):

    def __init__(self):
        super(PointCNN, self).__init__()

        self.conv1 = tg.XConv(in_channels=3, out_channels=256, dim=3, kernel_size=8, hidden_channels=128)
        self.conv2 = tg.XConv(in_channels=256, out_channels=512, dim=3, kernel_size=12)
        self.conv3 = tg.XConv(in_channels=512, out_channels=768, dim=3, kernel_size=16)
        # self.conv4 = tg.XConv(in_channels=768, out_channels=1024, dim=3, kernel_size=16)
        #
        # self.dconv1 = tg.XConv(in_channels=1024, out_channels=1024, dim=3, kernel_size=16)
        self.dconv2 = tg.XConv(in_channels=768, out_channels=768, dim=3, kernel_size=16)
        self.dconv3 = tg.XConv(in_channels=768, out_channels=512, dim=3, kernel_size=12)
        self.dconv4 = tg.XConv(in_channels=512, out_channels=256, dim=3, kernel_size=8)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=5),
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):
        pts = x.view(-1, 3).float()
        fts = x.view(-1, 3).float()
        x = self.conv1(fts, pts)
        x = self.conv2(x, pts)
        x = self.conv3(x, pts)
        # x = self.conv4(x, pts)
        # x = self.dconv1(x, pts)
        x = self.dconv2(x, pts)
        x = self.dconv3(x, pts)
        x = self.dconv4(x, pts)
        logit = self.fc_layer(x)
        return logit

    def training_step(self, batch, batch_idx):
        x, y = batch['pcd'], batch['label']
        y_hat = self(x)
        y_hat = y_hat.view(y.size()[0], y.size()[1], -1).type(torch.float)
        y_one_hot = F.one_hot(y, num_classes=5).type(torch.float)
        loss = F.cross_entropy(y_hat, y_one_hot)
        pred = F.softmax(y_hat, dim=-1)
        pred_idxs = torch.argmax(pred, dim=-1)
        self.accuracy(pred_idxs, y)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
