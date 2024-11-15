import os
import time
from tqdm import tqdm

import torch
from torch import nn
import numpy as np
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    MulticlassJaccardIndex)
import wandb

from dataloader.mfinstseg import MFInstSegDataset
from models.inst_segmentors_byChb import SFRGNNSegmentor
from utils.misc import seed_torch, init_logger, print_num_params
from collections import Counter
import torch.nn.functional as F
# 定义计算类别权重的函数
def compute_class_weights(train_dataset, num_classes):
    class_counts = Counter()

    # 遍历训练集，统计每个类别的样本数量
    for data in train_dataset:
        graphs = data["graph"]
        seg_labels = graphs.ndata["seg_y"]
        class_counts.update(seg_labels.tolist())

    # 计算总样本数
    total_count = sum(class_counts.values())

    # 计算每个类别的权重
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}

    # 标准化权重，使其最大值为1
    max_weight = max(class_weights.values())
    normalized_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    # 将权重转换为Tensor
    weights = torch.zeros(num_classes)
    for cls, weight in normalized_weights.items():
        weights[cls] = weight
    return weights

# 定义Balanced Focal Loss
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
        else:
            alpha_t = 1.0

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 定义IoU Loss
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        if len(targets.size()) == 1:  # 如果target是1D的
            targets = targets.unsqueeze(0)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).float()
        if len(targets_one_hot.size()) == 2:  # 如果one_hot是2D的
            targets_one_hot = targets_one_hot.unsqueeze(0)
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2))
        union = (inputs + targets_one_hot).sum(dim=(0, 2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou.mean()


# 合并的损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.focal_loss = BalancedFocalLoss(alpha, gamma, reduction)
        self.iou_loss = IoULoss()

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        iou_loss = self.iou_loss(inputs, targets)
        return focal_loss + iou_loss


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    os.environ["WANDB_API_KEY"] = 'xxxxxx'
    os.environ["WANDB_MODE"] = "offline"

    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
        project="sfrgnn" + "MFInstSegMultiTask",
        config={
            "edge_attr_dim": 12,
            "node_attr_dim": 10,
            "edge_attr_emb": 64,#原始64
            "node_attr_emb": 64,#原始64
            "edge_grid_dim": 0,
            "node_grid_dim": 7,
            "edge_grid_emb": 0,
            "node_grid_emb": 64,#原始64
            "num_layers": 2,
            "delta": 2,
            "mlp_ratio": 2,
            "drop": 0.25,
            "drop_path": 0.25,
            "head_hidden_dim": 64,
            "conv_on_edge": False,
            "use_uv_gird": True,
            "use_edge_attr": True,
            "use_face_attr": True,

            "seed": 42,
            "device": 'cuda',
            "architecture": "SFRGraphEncoder",
            "dataset_type": "full",
            "dataset": "/mnt/data",

            "epochs": 100,
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "batch_size": 256,
            "ema_decay_per_epoch": 1. / 2.,
            "seg_a": 1.,
            "inst_a": 1.,
            "bottom_a": 1.,
        }
    )

    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = wandb.config['device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = wandb.config['dataset']
    dataset_type = wandb.config['dataset_type']



    n_classes = 25

    model = SFRGNNSegmentor(num_classes=n_classes,
                            arch=wandb.config['architecture'],
                            edge_attr_dim=wandb.config['edge_attr_dim'],
                            node_attr_dim=wandb.config['node_attr_dim'],
                            edge_attr_emb=wandb.config['edge_attr_emb'],
                            node_attr_emb=wandb.config['node_attr_emb'],
                            edge_grid_dim=wandb.config['edge_grid_dim'],
                            node_grid_dim=wandb.config['node_grid_dim'],
                            edge_grid_emb=wandb.config['edge_grid_emb'],
                            node_grid_emb=wandb.config['node_grid_emb'],
                            num_layers=wandb.config['num_layers'],
                            delta=wandb.config['delta'],
                            mlp_ratio=wandb.config['mlp_ratio'],
                            drop=wandb.config['drop'],
                            drop_path=wandb.config['drop_path'],
                            head_hidden_dim=wandb.config['head_hidden_dim'],
                            conv_on_edge=wandb.config['conv_on_edge'],
                            use_uv_gird=wandb.config['use_uv_gird'],
                            use_edge_attr=wandb.config['use_edge_attr'],
                            use_face_attr=wandb.config['use_face_attr'])
                            # dim_node=256#域自适应训练需添加该参数
    model = model.to(device)

    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params

    train_dataset = MFInstSegDataset(root_dir=dataset, split='train',
                                     center_and_scale=False, normalize=True, random_rotate=False,
                                     dataset_type=dataset_type, num_threads=8)
    # no need to load graphs again !
    graphs = train_dataset.graphs()
    val_dataset = MFInstSegDataset(root_dir=dataset, graphs=graphs, split='val',
                                   center_and_scale=False, normalize=True,
                                   dataset_type=dataset_type, num_threads=8)

    train_loader = train_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)
    val_loader = val_dataset.get_dataloader(batch_size=wandb.config['batch_size'], shuffle=False, drop_last=False,
                                            pin_memory=True)



    class_weights = compute_class_weights(train_dataset, num_classes=n_classes)#计算类别权重
    class_weights = class_weights.to(device)

    seg_loss = CombinedLoss(alpha=class_weights, gamma=2)
    inst_loss = nn.BCEWithLogitsLoss()
    bottom_loss = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=wandb.config['epochs'], eta_min=0)

    train_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    train_inst_acc = BinaryAccuracy().to(device)
    train_bottom_acc = BinaryAccuracy().to(device)

    train_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    train_inst_f1 = BinaryF1Score().to(device)
    train_bottom_iou = BinaryJaccardIndex().to(device)

    val_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    val_inst_acc = BinaryAccuracy().to(device)
    val_bottom_acc = BinaryAccuracy().to(device)

    val_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    val_inst_f1 = BinaryF1Score().to(device)
    val_bottom_iou = BinaryJaccardIndex().to(device)

    iters = len(train_loader)
    ema_decay = wandb.config['ema_decay_per_epoch'] ** (1 / iters)
    print(f'EMA decay: {ema_decay}')
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    best_acc = 0.
    save_path = 'output'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = init_logger(os.path.join(save_path, 'log.txt'))

    for epoch in range(wandb.config['epochs']):
        logger.info(f'------------- Now start epoch {epoch}------------- ')
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader)

        for data in train_bar:

            graphs = data["graph"].to(device, non_blocking=True)#一个批次的所有图
            # 维度：len(batch) * max_num_nodes * max_num_nodes #一个批次数据的inst标签
            inst_label = data["inst_labels"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]
            bottom_label = graphs.ndata["bottom_y"]

            opt.zero_grad(set_to_none=True)

            seg_pred, inst_pred, bottom_pred = model(graphs)


            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = inst_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)

            loss = wandb.config['seg_a'] * loss_seg + \
                   wandb.config['inst_a'] * loss_inst + \
                   wandb.config['bottom_a'] * loss_bottom

            train_losses.append(loss.item())

            lr = opt.param_groups[0]["lr"]
            info = "Epoch:%d LR:%f Seg:%f Inst:%f Bottom:%f  Total:%f" % (
                epoch, lr, loss_seg, loss_inst, loss_bottom,  loss)
            train_bar.set_description(info)

            loss.backward()
            opt.step()
            ema.update()

            train_seg_acc.update(seg_pred, seg_label)
            train_seg_iou.update(seg_pred, seg_label)
            train_inst_acc.update(inst_pred, inst_label)
            train_inst_f1.update(inst_pred, inst_label)
            train_bottom_acc.update(bottom_pred, bottom_label)
            train_bottom_iou.update(bottom_pred, bottom_label)

        scheduler.step()
        mean_train_loss = np.mean(train_losses).item()
        mean_train_seg_acc = train_seg_acc.compute().item()
        mean_train_seg_iou = train_seg_iou.compute().item()
        mean_train_inst_acc = train_inst_acc.compute().item()
        mean_train_inst_f1 = train_inst_f1.compute().item()
        mean_train_bottom_acc = train_bottom_acc.compute().item()
        mean_train_bottom_iou = train_bottom_iou.compute().item()

        logger.info(f'train_loss : {mean_train_loss}, \
                      train_seg_acc: {mean_train_seg_acc}, \
                      train_seg_iou: {mean_train_seg_iou}, \
                      train_inst_acc: {mean_train_inst_acc}, \
                      train_inst_f1: {mean_train_inst_f1}, \
                      train_bottom_acc: {mean_train_bottom_acc}, \
                      train_bottom_iou: {mean_train_bottom_iou}')
        wandb.log({'epoch': epoch,
                   'train_loss': mean_train_loss,
                   'train_seg_acc': mean_train_seg_acc,
                   'train_seg_iou': mean_train_seg_iou,
                   'train_inst_acc': mean_train_inst_acc,
                   'train_inst_f1': mean_train_inst_f1,
                   'train_bottom_acc': mean_train_bottom_acc,
                   'train_bottom_iou': mean_train_bottom_iou})

        train_seg_acc.reset()
        train_inst_acc.reset()
        train_bottom_acc.reset()
        train_seg_iou.reset()
        train_inst_f1.reset()
        train_bottom_iou.reset()

        with torch.no_grad():
            with ema.average_parameters():
                model.eval()
                val_losses = []
                for data in tqdm(val_loader):
                    graphs = data["graph"].to(device)
                    inst_label = data["inst_labels"].to(device)
                    seg_label = graphs.ndata["seg_y"]
                    bottom_label = graphs.ndata["bottom_y"]

                    with torch.autocast(device_type= "cuda" if torch.cuda.is_available() else "cpu",
                                        dtype=torch.float16):
                        seg_pred, inst_pred, bottom_pred = model(graphs)

                        loss_seg = seg_loss(seg_pred, seg_label)
                        loss_inst = inst_loss(inst_pred, inst_label)
                        loss_bottom = bottom_loss(bottom_pred, bottom_label)

                        loss = wandb.config['seg_a'] * loss_seg + \
                               wandb.config['inst_a'] * loss_inst + \
                               wandb.config['bottom_a'] * loss_bottom


                    val_losses.append(loss.item())

                    val_seg_acc.update(seg_pred, seg_label)
                    val_seg_iou.update(seg_pred, seg_label)
                    val_inst_acc.update(inst_pred, inst_label)
                    val_inst_f1.update(inst_pred, inst_label)
                    val_bottom_acc.update(bottom_pred, bottom_label)
                    val_bottom_iou.update(bottom_pred, bottom_label)
                mean_val_loss = np.mean(val_losses).item()
                mean_val_seg_acc = val_seg_acc.compute().item()
                mean_val_seg_iou = val_seg_iou.compute().item()
                mean_val_inst_acc = val_inst_acc.compute().item()
                mean_val_inst_f1 = val_inst_f1.compute().item()
                mean_val_bottom_acc = val_bottom_acc.compute().item()
                mean_val_bottom_iou = val_bottom_iou.compute().item()

                logger.info(f'val_loss : {mean_val_loss}, \
                              val_seg_acc: {mean_val_seg_acc}, \
                              val_seg_iou: {mean_val_seg_iou}, \
                              val_inst_acc: {mean_val_inst_acc}, \
                              val_inst_f1: {mean_val_inst_f1}, \
                              val_bottom_acc: {mean_val_bottom_acc}, \
                              val_bottom_iou: {mean_val_bottom_iou}')
                wandb.log({'epoch': epoch,
                           'val_loss': mean_val_loss,
                           'val_seg_acc': mean_val_seg_acc,
                           'val_seg_iou': mean_val_seg_iou,
                           'val_inst_acc': mean_val_inst_acc,
                           'val_inst_f1': mean_val_inst_f1,
                           'val_bottom_acc': mean_val_bottom_acc,
                           'val_bottom_iou': mean_val_bottom_iou
                         })

                val_seg_acc.reset()
                val_seg_iou.reset()
                val_inst_acc.reset()
                val_inst_f1.reset()
                val_bottom_acc.reset()
                val_bottom_iou.reset()

                cur_acc = mean_val_seg_iou + mean_val_inst_f1 + mean_val_bottom_iou
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    logger.info(f'best metric: {cur_acc}, model saved')
                    torch.save(model.state_dict(), os.path.join(save_path, "weight_%d-epoch.pth" % (epoch)))

    graphs = train_dataset.graphs()
    test_dataset = MFInstSegDataset(root_dir=dataset, graphs=graphs, split='test',
                                    center_and_scale=False, normalize=True, random_rotate=False,
                                    dataset_type=dataset_type, num_threads=8)
    test_loader = test_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_inst_acc = BinaryAccuracy().to(device)
    test_bottom_acc = BinaryAccuracy().to(device)

    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    test_inst_f1 = BinaryF1Score().to(device)
    test_bottom_iou = BinaryJaccardIndex().to(device)

    with torch.no_grad():
        logger.info(f'------------- Now start testing ------------- ')
        model.eval()
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            inst_label = data["inst_labels"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]
            bottom_label = graphs.ndata["bottom_y"]

            seg_pred, inst_pred, bottom_pred = model(graphs)


            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = inst_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)
            loss = wandb.config['seg_a'] * loss_seg + \
                   wandb.config['inst_a'] * loss_inst + \
                   wandb.config['bottom_a'] * loss_bottom

            test_losses.append(loss.item())
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            test_inst_acc.update(inst_pred, inst_label)
            test_inst_f1.update(inst_pred, inst_label)
            test_bottom_acc.update(bottom_pred, bottom_label)
            test_bottom_iou.update(bottom_pred, bottom_label)

        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        mean_test_inst_acc = test_inst_acc.compute().item()
        mean_test_inst_f1 = test_inst_f1.compute().item()
        mean_test_bottom_acc = test_bottom_acc.compute().item()
        mean_test_bottom_iou = test_bottom_iou.compute().item()

        logger.info(f'test_loss : {mean_test_loss}, \
                      test_seg_acc: {mean_test_seg_acc}, \
                      test_seg_iou: {mean_test_seg_iou}, \
                      test_inst_acc: {mean_test_inst_acc}, \
                      test_inst_f1: {mean_test_inst_f1}, \
                      test_bottom_acc: {mean_test_bottom_acc}, \
                      test_bottom_iou: {mean_test_bottom_iou}')
        wandb.log({'test_loss': mean_test_loss,
                   'test_seg_acc': mean_test_seg_acc,
                   'test_seg_iou': mean_test_seg_iou,
                   'test_inst_acc': mean_test_inst_acc,
                   'test_inst_f1': mean_test_inst_f1,
                   'test_bottom_acc': mean_test_bottom_acc,
                   'test_bottom_iou': mean_test_bottom_iou})
