from typing import List, Union

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from model.utils.iou import *
from torch_geometric.nn import XConv, fps, knn_interpolate
from pytorch_lightning import LightningModule


def down_sample_layer(x, pose, batch, ratio=0.375):
    idx = fps(pose, batch, ratio=ratio)
    x, pose, batch = x[idx], pose[idx], batch[idx]
    return x, pose, batch


class PointCNNSeg(LightningModule):

    def __init__(self, num_classes, weight_balance=None):
        super(PointCNNSeg, self).__init__()

        self.num_classes = num_classes
        if weight_balance is not None:
            self.weight_balance = torch.tensor(weight_balance)
        else:
            self.weight_balance = weight_balance
        layers_D = [64, 128, 256, 512]
        hidden_D = [int((0 + layers_D[0]) / 2), int((layers_D[0] + layers_D[1]) / 2),
                    int((layers_D[1] + layers_D[2]) / 2), int((layers_D[2] + layers_D[3]) / 2)]

        layers_U = [512, 256, 128, 64]
        hidden_U = [int((layers_U[0] + layers_U[1]) / 2), int((layers_U[1] + layers_U[2]) / 2),
                    int((layers_U[2] + layers_U[3]) / 2), int((layers_U[3] + layers_U[3]) / 2)]

        self.conv1 = XConv(0, layers_D[0], dim=3, kernel_size=8, dilation=1, hidden_channels=hidden_D[0])
        self.conv2 = XConv(layers_D[0], layers_D[1], dim=3, kernel_size=12, hidden_channels=hidden_D[1], dilation=2)
        self.conv3 = XConv(layers_D[1], layers_D[2], dim=3, kernel_size=16, hidden_channels=hidden_D[2], dilation=2)
        self.conv4 = XConv(layers_D[2], layers_D[3], dim=3, kernel_size=16, hidden_channels=hidden_D[3], dilation=4)

        self.conv_up4 = XConv(layers_U[0], layers_U[0], dim=3, kernel_size=16, hidden_channels=hidden_U[0], dilation=4)
        self.conv_up3 = XConv(layers_U[0], layers_U[1], dim=3, kernel_size=16, hidden_channels=hidden_U[1], dilation=2)
        self.conv_up2 = XConv(layers_U[1], layers_U[2], dim=3, kernel_size=12, hidden_channels=hidden_U[2], dilation=2)
        self.conv_up1 = XConv(layers_U[2], layers_U[3], dim=3, kernel_size=8, hidden_channels=hidden_U[3], dilation=2)

        self.mlp_out4 = nn.Conv1d(layers_U[0], layers_U[0], kernel_size=1)
        self.mlp_out3 = nn.Conv1d(layers_U[1], layers_U[1], kernel_size=1)
        self.mlp_out2 = nn.Conv1d(layers_U[2], layers_U[2], kernel_size=1)
        self.mlp_out1 = nn.Conv1d(layers_U[3], layers_U[3], kernel_size=1)

        self.down_sampler = down_sample_layer

        ##head because of in pytorch geometric we can't use batch dimention like [1,num_classes,number_of_point] we cant use nn.sequential
        self.fc_layer1 = nn.Conv1d(layers_U[3], 128, kernel_size=1, bias=False)
        self.Relu = nn.ReLU(True)
        self.DROP = nn.Dropout(0.5)
        self.fc_layer2 = nn.Conv1d(128, self.num_classes, kernel_size=1)

        self.batch_size = 16
        self.number_of_point = 2048
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuary = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    # pytorch geometric layer need batch and pos as input batch is like batch = [1,1,2,2,3,3] means firest data is batch 1
    # and second data is batch 2 and so on


    def pre_pointcnn(self, points):

        self.batch_size = points.shape[0]
        self.number_of_point = points.shape[1]

        batch_zero = torch.zeros(points[0].shape[0], dtype=torch.int64)
        batch = torch.zeros(points[0].shape[0], dtype=torch.int64)
        point_for_pointcnn = points[0]
        for b in range(1, points.shape[0]):
            batch = torch.cat((batch, batch_zero + b), dim=0)
            point_for_pointcnn = torch.cat((point_for_pointcnn, points[b]), dim=0)
        points = point_for_pointcnn
        points = points.to(self.device)
        batch = batch.to(self.device)

        return points, batch

    # after predict we need to reshape the output to be like [batch,num_classes,number_of_point] beacuse of pytorch geometric layer condition
    # see pre_pointcnn comment for more information

    def after_pred(self, preds, batch):

        out_batch = torch.zeros(self.batch_size, self.num_classes, self.number_of_point)
        out = preds.T

        for b in range(self.batch_size):
            out_batch[b, :, :] = out[batch == b].T
        preds = out_batch
        preds = preds.to(self.device)
        return preds

    def forward(self, points):
        pos0, batch0 = self.pre_pointcnn(points)

        pos1, batch1 = pos0.type(torch.FloatTensor).to(self.device), batch0
        # print(pos1.shape)
        x1 = F.relu(self.conv1(None, pos1, batch1))

        x2, pos2, batch2 = self.down_sampler(x1, pos1, batch1)
        # print(x2.shape)
        x2 = F.relu(self.conv2(x2, pos2, batch2))

        x3, pos3, batch3 = self.down_sampler(x2, pos2, batch2, ratio=0.375)
        # print(x3.shape)
        x3 = F.relu(self.conv3(x3, pos3, batch3))

        x4, pos4, batch4 = self.down_sampler(x3, pos3, batch3, ratio=0.375)
        # print(x4.shape)
        x4 = F.relu(self.conv4(x4, pos4, batch4))

        xo4 = F.relu(self.conv_up4(x4, pos4, batch4))

        xo4_concat = (xo4 + x4).T
        xo4_after_mlp = self.mlp_out4(xo4_concat).T
        Xo3_in = knn_interpolate(x=xo4_after_mlp, pos_x=pos4, batch_x=batch4, k=3, pos_y=pos3, batch_y=batch3)
        xo3 = F.relu(self.conv_up3(Xo3_in, pos3, batch3))

        xo3_concat = (xo3 + x3).T
        xo3_after_mlp = self.mlp_out3(xo3_concat).T
        Xo2_in = knn_interpolate(x=xo3_after_mlp, pos_x=pos3, batch_x=batch3, k=3, pos_y=pos2, batch_y=batch2)

        xo2 = F.relu(self.conv_up2(Xo2_in, pos2, batch2))

        xo2_concat = (xo2 + x2).T

        xo2_after_mlp = self.mlp_out2(xo2_concat).T

        Xo1_in = knn_interpolate(x=xo2_after_mlp, pos_x=pos2, batch_x=batch2, k=3, pos_y=pos1, batch_y=batch1)

        xo1 = F.relu(self.conv_up1(Xo1_in, pos1, batch1))

        xo1_concat = (xo1 + x1).T

        xo1_after_mlp = self.mlp_out1(xo1_concat)

        X_OUT = torch.unsqueeze(xo1_after_mlp.T, 0)
        # X_OUT = self.BN(X_OUT)

        X_OUT = self.fc_layer1(xo1_after_mlp)
        X_OUT = self.Relu(X_OUT)
        X_OUT = self.DROP(X_OUT)
        X_OUT = self.fc_layer2(X_OUT)

        X_OUT = self.after_pred(X_OUT, batch=batch0)

        return X_OUT

    def training_step(self, batch, batch_idx):
        x, y = batch['pcd'], batch['label']
        y_hat = self(x)
        y_hat = y_hat.type(torch.float)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).type(torch.float).permute((0, 2, 1))
        if self.weight_balance is None:
            loss = F.cross_entropy(y_hat, y_one_hot)
        else:
            loss = F.cross_entropy(y_hat, y_one_hot, weight=self.weight_balance.type(torch.float32).to(self.device), ignore_index=-1)
        pred = F.softmax(y_hat.permute((0, 2, 1)), dim=-1)
        pred_idxs = torch.argmax(pred, dim=-1)
        intersection, union, target = intersection_union_gpu(pred_idxs, y, self.num_classes)
        iou_class = intersection / (union + 1e-10)
        iou = torch.sum(intersection) / torch.sum(union + 1e-10)
        m_iou = torch.mean(iou)
        self.train_accuracy.update(pred_idxs, y)
        # for i in range(self.num_classes):
        #    self.log(f'train_iou_class_{i}_step', iou_class[i])
        # self.log('train_mIoU_step', m_iou)
        # self.log('train_loss_step', loss)
        # self.log('train_acc_step', self.train_accuracy)
        return {'loss': loss, 'train_mIoU': m_iou, 'train_acc_step': self.train_accuracy, 'train_class_iou': iou_class}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        result = {'loss': 0, 'train_mIoU': 0, 'train_acc_step': 0, 'train_class_iou': torch.zeros(self.num_classes).to(self.device)}
        for item in outputs:
            for k, v in item.items():
                if k != 'train_acc_step':
                    result[k] += v
        for k, v in result.items():
            if k != 'train_acc_step':
                result[k] = result[k] / len(outputs)
        for i in range(self.num_classes):
            self.log(f'train_iou_class_{i}_epoch', result['train_class_iou'][i])
        self.log('train_mIoU_epoch', result['train_mIoU'], on_step=False, on_epoch=True)
        self.log('train_loss_epoch', result['loss'], on_step=False, on_epoch=True)
        self.log('train_acc_epoch', outputs[-2]['train_acc_step'], on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch['pcd'], batch['label']
        y_hat = self(x)
        y_hat = y_hat.type(torch.float)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).type(torch.float).permute((0, 2, 1))
        loss = F.cross_entropy(y_hat, y_one_hot)
        pred = F.softmax(y_hat.permute((0, 2, 1)), dim=-1)
        pred_idxs = torch.argmax(pred, dim=-1)
        intersection, union, target = intersection_union_gpu(pred_idxs, y, self.num_classes)
        iou_class = intersection / (union + 1e-10)
        iou = torch.sum(intersection) / torch.sum(union + 1e-10)
        m_iou = torch.mean(iou)
        self.train_accuracy.update(pred_idxs, y)
        return {'loss': loss, 'val_mIoU': m_iou, 'val_acc_step': self.train_accuracy, 'val_class_iou': iou_class}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        result = {'loss': 0, 'val_mIoU': 0, 'val_acc_step': 0, 'val_class_iou': torch.zeros(self.num_classes).to(self.device)}
        for item in outputs:
            for k, v in item.items():
                if k != 'val_acc_step':
                    result[k] += v
        for k, v in result.items():
            if k != 'val_acc_step':
                result[k] = result[k] / len(outputs)
        for i in range(self.num_classes):
            self.log(f'val_iou_class_{i}_epoch', result['val_class_iou'][i])
        self.log('val_mIoU_epoch', result['val_mIoU'], on_step=False, on_epoch=True)
        self.log('val_loss_epoch', result['loss'], on_step=False, on_epoch=True)
        self.log('val_acc_epoch', outputs[-2]['val_acc_step'], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['pcd'], batch['label']
        y_hat = self(x)
        y_hat = y_hat.view(y.size()[0], y.size()[1], -1).type(torch.float)
        y_one_hot = F.one_hot(y, num_classes=5).type(torch.float)
        loss = F.cross_entropy(y_hat, y_one_hot)
        pred = F.softmax(y_hat, dim=-1)
        pred_idxs = torch.argmax(pred, dim=-1)
        self.test_acc.update(pred_idxs, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['pcd'], batch['label']
        y_hat = self(x)
        y_hat = y_hat.view(y.size()[0], y.size()[1], -1).type(torch.float)
        pred = F.softmax(y_hat, dim=-1)
        pred_idxs = torch.argmax(pred, dim=-1)
        return pred_idxs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(1e-3))
        return optimizer
