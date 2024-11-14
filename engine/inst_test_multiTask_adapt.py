
from torchmetrics.classification import (
    MulticlassAccuracy,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    MulticlassJaccardIndex,
    BinaryAveragePrecision)



from tqdm import tqdm
import torch
from torch import nn
import numpy as np




from dataloader.mfinstseg import MFInstSegDataset
from models.inst_segmentors_byChb_adapt_multiSelect_v1 import SFRGNNSegmentor

from utils.misc import seed_torch
import torch.nn.functional as F



class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
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
        # 计算每个类别的概率
        inputs = torch.softmax(inputs, dim=1)  # (batch_size, num_classes)

        # 将targets转换为one-hot编码
        targets_one_hot = torch.nn.functional.one_hot(targets,
                                                      num_classes=inputs.shape[1]).float()  # (batch_size, num_classes)

        # 计算交集和并集
        intersection = (inputs * targets_one_hot).sum(dim=0)  # (num_classes,)
        union = (inputs + targets_one_hot).sum(dim=0) - intersection  # (num_classes,)

        # 计算IoU
        iou = (intersection + 1e-6) / (union + 1e-6)  # (num_classes,)

        # 返回平均IoU损失
        return 1 - iou.mean()


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
    # track hyperparameters and run metadata
    torch.set_float32_matmul_precision("high") # may be faster if GPU support TF32
    config={
        "edge_attr_dim": 12,
        "node_attr_dim": 10,
        "edge_attr_emb": 64, # recommend: 64
        "node_attr_emb": 64, # recommend: 64
        "edge_grid_dim": 0,
        "node_grid_dim": 7,
        "edge_grid_emb": 0,
        "node_grid_emb": 64, # recommend: 64
        "num_layers": 2, # recommend: 2
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
        "dataset": "/path/dataset",#
        "adaptation_method": "madann",  # dann, mmd, tca

        "epochs": 100,
        "lr": 1e-2,
        "weight_decay": 1e-2,
        "batch_size": 256,
        "ema_decay_per_epoch": 1. / 2.,
        "seg_a": 1.,
        "inst_a": 1.,
        "bottom_a": 1.,
        }

    seed_torch(config['seed'])
    device = config['device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("GPU is enabled.")
    else:
        print("GPU is not enabled. Using CPU instead.")

    n_classes = 25

    model = SFRGNNSegmentor(num_classes=n_classes,
                            arch=config['architecture'],
                            edge_attr_dim=config['edge_attr_dim'],
                            node_attr_dim=config['node_attr_dim'],
                            edge_attr_emb=config['edge_attr_emb'],
                            node_attr_emb=config['node_attr_emb'],
                            edge_grid_dim=config['edge_grid_dim'],
                            node_grid_dim=config['node_grid_dim'],
                            edge_grid_emb=config['edge_grid_emb'],
                            node_grid_emb=config['node_grid_emb'],
                            num_layers=config['num_layers'],
                            delta=config['delta'],
                            mlp_ratio=config['mlp_ratio'],
                            drop=config['drop'],
                            drop_path=config['drop_path'],
                            head_hidden_dim=config['head_hidden_dim'],
                            conv_on_edge=config['conv_on_edge'],
                            use_uv_gird=config['use_uv_gird'],
                            use_edge_attr=config['use_edge_attr'],
                            use_face_attr=config['use_face_attr'],
                            adaptation_method=config['adaptation_method'])
    model = model.to(device)
    #
    model_param = torch.load("/path/weight",map_location=device)
    model.load_state_dict(model_param)
    dataset = config['dataset']

    test_dataset = MFInstSegDataset(root_dir=dataset, split='test', nums_data=3400,
                                     center_and_scale=False, normalize=True, random_rotate=True,
                                     num_threads=8)

    test_loader = test_dataset.get_dataloader(batch_size=config['batch_size'], pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()
    instance_loss = nn.BCEWithLogitsLoss()
    bottom_loss = nn.BCEWithLogitsLoss()

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_inst_acc = BinaryAccuracy().to(device)
    test_bottom_acc = BinaryAccuracy().to(device)

    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    test_inst_f1 = BinaryF1Score().to(device)
    test_bottom_iou = BinaryJaccardIndex().to(device)

    best_acc = 0.
    with torch.no_grad():
        print(f'------------- Now start testing ------------- ')
        model.eval()
        # test_per_inst_acc = []
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            inst_label = data["inst_labels"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]
            bottom_label = graphs.ndata["bottom_y"]


            # Forward pass
            seg_pred, inst_pred, bottom_pred, _ = model(graphs)

            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = instance_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)
            loss = config['seg_a'] * loss_seg + \
                   config['inst_a'] * loss_inst + \
                   config['bottom_a'] * loss_bottom
            test_losses.append(loss.item())
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            test_inst_acc.update(inst_pred, inst_label)
            test_inst_f1.update(inst_pred, inst_label)
            test_bottom_acc.update(bottom_pred, bottom_label)
            test_bottom_iou.update(bottom_pred, bottom_label)

        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        mean_test_inst_acc = test_inst_acc.compute().item()
        mean_test_inst_f1 = test_inst_f1.compute().item()
        mean_test_bottom_acc = test_bottom_acc.compute().item()
        mean_test_bottom_iou = test_bottom_iou.compute().item()

        print(f'test_loss : {mean_test_loss}, \
                 test_seg_acc: {mean_test_seg_acc}, \
                 test_seg_iou: {mean_test_seg_iou}, \
                 test_inst_acc: {mean_test_inst_acc}, \
                 test_inst_f1: {mean_test_inst_f1}, \
                 test_bottom_acc: {mean_test_bottom_acc}, \
                 test_bottom_iou: {mean_test_bottom_iou}')
