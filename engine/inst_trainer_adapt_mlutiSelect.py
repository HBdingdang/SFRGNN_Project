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
    MulticlassJaccardIndex
)
import wandb
from dataloader.mfinstseg_bychb_v2 import MFInstSegAdaptiveDataset
from models.inst_segmentors_byChb_adapt_multiSelect_v1 import SFRGNNSegmentor
from models.loss import entropy_loss, mmd_loss, tca_loss
from utils.misc import seed_torch, init_logger, print_num_params
import torch.nn.functional as F
from collections import Counter

# 定义计算类别权重的函数，仅基于源域数据
def compute_class_weights(train_dataset, num_classes, device):
    class_counts = Counter()

    # 遍历训练集，统计源域每个类别的样本数量
    for data in tqdm(train_dataset, desc="Computing class weights"):
        # 访问 source_data 中的 graph
        source_data = data["source_data"]
        if "graph" not in source_data:
            print("Warning: 'graph' key not found in source_data")
            continue

        source_graphs = source_data["graph"]
        if "seg_y" not in source_graphs.ndata:
            print("Warning: 'seg_y' key not found in graph.ndata")
            continue

        seg_labels = source_graphs.ndata["seg_y"].cpu().numpy()
        class_counts.update(seg_labels.tolist())

    # 计算总样本数
    total_count = sum(class_counts.values())
    if total_count == 0:
        raise ValueError("No samples found in source domain for computing class weights.")

    # 计算每个类别的权重，避免除以零
    class_weights = {cls: total_count / count for cls, count in class_counts.items() if count > 0}

    # 标准化权重，使其最大值为1
    max_weight = max(class_weights.values())
    normalized_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    # 将权重转换为Tensor
    weights = torch.zeros(num_classes, device=device)
    for cls, weight in normalized_weights.items():
        if cls < num_classes:
            weights[cls] = weight
        else:
            print(f"Warning: Class index {cls} exceeds num_classes {num_classes}")

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
    os.environ["WANDB_API_KEY"] = 'your_wandb_api_key'
    os.environ["WANDB_MODE"] = "offline"

    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
        project="domain_adapt_methods_madann" + "smallMFInst-MFInst",
        config={
            "edge_attr_dim": 12,
            "node_attr_dim": 10,
            "edge_attr_emb": 64,
            "node_attr_emb": 64,
            "edge_grid_dim": 0,
            "node_grid_dim": 7,
            "edge_grid_emb": 0,
            "node_grid_emb": 64,
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
            "dataset_source": "/path/source_dataset",
            "dataset_target": "/path/target_dataset",
            "train_data_nums": 16000,
            "val_data_nums": 3400,
            "test_target_data_nums": 3400,
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 1e-2,
            "batch_size": 256,
            "ema_decay_per_epoch": 1. / 2.,
            "seg_a": 5,
            "inst_a": 3,
            "bottom_a": 1,
            "adaptation_method": "madann",
            "mmd_weight": 0.1,
            "tca_weight": 0.1,
            "dann_weight": 0.008,
            "lr_g": 0.01,
            "lr_d": 0.01,
            "d_steps": 1,  # 域判别器的训练步数
            "lambda_gp": 10.0
        }
    )

    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    dataset_type = wandb.config['dataset_type']
    n_classes = 25

    # Initialize the model
    model = SFRGNNSegmentor(
        num_classes=n_classes,
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
        use_face_attr=wandb.config['use_face_attr'],
        dim_node=256,
        adaptation_method=wandb.config['adaptation_method']
    ).to(device)

    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params

    source_dir = wandb.config['dataset_source']
    target_dir = wandb.config['dataset_target']

    train_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        split='train',
        center_and_scale=False,
        normalize=True,
        random_rotate=True,
        num_threads=8,
        nums_data=wandb.config['train_data_nums']
    )
    print(f"Total train dataset size: {len(train_dataset)}")
    print(f"Source data size: {len(train_dataset.data_source)}, Target data size: {len(train_dataset.data_target)}")

    train_loader = train_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    val_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        split='val',
        center_and_scale=False,
        normalize=True,
        random_rotate=True,
        num_threads=8,
        nums_data=wandb.config['val_data_nums']
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    val_loader = val_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    class_weights = compute_class_weights(train_dataset, num_classes=n_classes, device=device)
    print(f"Class weights: {class_weights}")

    # �I_1�p
    combined_loss_fn = CombinedLoss(alpha=class_weights)

    # Metrics setup
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

    # 定义优化器和调度器
    if wandb.config['adaptation_method'] in ['madann', 'dann']:
        feature_extractor_params = []
        for name, param in model.named_parameters():
            if 'domain_discriminator' not in name:
                feature_extractor_params.append(param)
        domain_discriminator_params = model.domain_discriminator.parameters()

        # 对特征提取器和判别器使用不同的优化器和学习率
        optimizer_g = torch.optim.AdamW(feature_extractor_params, lr=wandb.config['lr_g'],
                                        weight_decay=wandb.config['weight_decay'])  # L2正则化
        optimizer_d = torch.optim.AdamW(domain_discriminator_params, lr=wandb.config['lr_d'],
                                        weight_decay=wandb.config['weight_decay'])  # L2正则化

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'],
                                      weight_decay=wandb.config['weight_decay'])  # L2正则化
        # 学习率调度器配置
    if wandb.config['adaptation_method'] in ['madann', 'dann']:
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=wandb.config['epochs'], eta_min=0)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=wandb.config['epochs'], eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb.config['epochs'], eta_min=0)


    iters = len(train_loader)
    ema_decay = wandb.config['ema_decay_per_epoch'] ** (1 / iters)
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    best_acc = 0.
    save_path = 'output'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, time_str)
    os.makedirs(save_path, exist_ok=True)
    logger = init_logger(os.path.join(save_path, 'log.txt'))

    # Training loop
    for epoch in range(wandb.config['epochs']):
        logger.info(f'------------- Now start epoch {epoch} -------------')
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for data in train_bar:
            if data is None:
                continue

            # 将数据转移到设备
            source_graphs = data["source_graph"].to(device, non_blocking=True)
            target_graphs = data["target_graph"].to(device, non_blocking=True)
            source_inst_label = data["source_inst_labels"].to(device, non_blocking=True)
            target_inst_label = data["target_inst_labels"].to(device, non_blocking=True)
            source_seg_label = source_graphs.ndata["seg_y"].to(device)
            target_seg_label = target_graphs.ndata["seg_y"].to(device)
            source_bottom_label = source_graphs.ndata["bottom_y"].to(device)
            target_bottom_label = target_graphs.ndata["bottom_y"].to(device)


            if wandb.config['adaptation_method'] == 'madann' or wandb.config['adaptation_method'] == 'dann':

                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                # 前向传播
                seg_pred_source, inst_pred_source, bottom_pred_source, node_emb_source = model(source_graphs)
                seg_pred_target, _, _, node_emb_target = model(target_graphs)

                # 源域任务损失
                loss_seg_source = combined_loss_fn(seg_pred_source, source_seg_label)
                loss_inst_source = nn.BCEWithLogitsLoss()(inst_pred_source, source_inst_label.float())
                loss_bottom_source = nn.BCEWithLogitsLoss()(bottom_pred_source, source_bottom_label.float())

                # 合并源域和目标域特征
                features = torch.cat((node_emb_source, node_emb_target), dim=0)
                # 域标签：源域为0，目标域为1
                domain_labels = torch.cat((
                    torch.zeros(node_emb_source.size(0)).to(device),
                    torch.ones(node_emb_target.size(0)).to(device)
                ), dim=0)

                # 通过GRL反转梯度
                from models.loss import grad_reverse

                features = grad_reverse(features)
                domain_output = model.domain_discriminator(features).squeeze()
                domain_loss = nn.BCEWithLogitsLoss()(domain_output, domain_labels)

                # Entropy loss for target domain
                seg_prob_t = F.softmax(seg_pred_target, dim=-1)
                loss_t = entropy_loss(seg_prob_t)

                # 总损失
                loss = (wandb.config['seg_a'] * loss_seg_source +
                        wandb.config['inst_a'] * loss_inst_source +
                        wandb.config['bottom_a'] * loss_bottom_source +
                        wandb.config['dann_weight'] * domain_loss
                        )
                # 总损失
                # loss = (wandb.config['seg_a'] * loss_seg_source +
                #         wandb.config['inst_a'] * loss_inst_source +
                #         wandb.config['bottom_a'] * loss_bottom_source +
                #         wandb.config['dann_weight'] * domain_loss +
                #         0.1 * loss_t)
                loss_adv = domain_loss
                # 反向传播和优化
                loss.backward()
                optimizer_g.step()
                optimizer_d.step()

            elif wandb.config['adaptation_method'] == 'mmd':
                # ============== MMD 训练步骤 ==============
                optimizer.zero_grad()

                # 前向传播
                seg_pred_source, inst_pred_source, bottom_pred_source, node_emb_source = model(source_graphs)
                seg_pred_target, _, _, node_emb_target = model(target_graphs)

                # 源域任务损失
                loss_seg_source = combined_loss_fn(seg_pred_source, source_seg_label)
                loss_inst_source = nn.BCEWithLogitsLoss()(inst_pred_source, source_inst_label.float())
                loss_bottom_source = nn.BCEWithLogitsLoss()(bottom_pred_source, source_bottom_label.float())

                # 计算MMD损失
                loss_mmd = mmd_loss(node_emb_source, node_emb_target)

                # Entropy loss for target domain
                seg_prob_t = F.softmax(seg_pred_target, dim=-1)
                loss_t = entropy_loss(seg_prob_t)

                # 总损失
                loss = (wandb.config['seg_a'] * loss_seg_source +
                        wandb.config['inst_a'] * loss_inst_source +
                        wandb.config['bottom_a'] * loss_bottom_source +
                        wandb.config['mmd_weight'] * loss_mmd +
                        0.1 * loss_t)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

            elif wandb.config['adaptation_method'] == 'tca':
                # ============== TCA 训练步骤 ==============
                optimizer.zero_grad()

                # 前向传播
                seg_pred_source, inst_pred_source, bottom_pred_source, node_emb_source = model(source_graphs)
                seg_pred_target, _, _, node_emb_target = model(target_graphs)

                # 源域任务损失
                loss_seg_source = combined_loss_fn(seg_pred_source, source_seg_label)
                loss_inst_source = nn.BCEWithLogitsLoss()(inst_pred_source, source_inst_label.float())
                loss_bottom_source = nn.BCEWithLogitsLoss()(bottom_pred_source, source_bottom_label.float())

                # 计算TCA损失
                loss_tca = tca_loss(node_emb_source, node_emb_target)

                # Entropy loss for target domain
                seg_prob_t = F.softmax(seg_pred_target, dim=-1)
                loss_t = entropy_loss(seg_prob_t)

                # 总损失
                loss = (wandb.config['seg_a'] * loss_seg_source +
                        wandb.config['inst_a'] * loss_inst_source +
                        wandb.config['bottom_a'] * loss_bottom_source +
                        wandb.config['tca_weight'] * loss_tca +
                        0.1 * loss_t)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

            else:
                raise ValueError(f"Unknown adaptation method: {wandb.config['adaptation_method']}")

            # EMA 更新
            ema.update()

            # 记录损失和指标
            train_losses.append(loss.item())
            train_seg_acc.update(F.softmax(seg_pred_source, dim=-1), source_seg_label)
            train_seg_iou.update(F.softmax(seg_pred_source, dim=-1), source_seg_label)
            train_inst_acc.update(torch.sigmoid(inst_pred_source), source_inst_label)
            train_inst_f1.update(torch.sigmoid(inst_pred_source), source_inst_label)
            train_bottom_acc.update(torch.sigmoid(bottom_pred_source), source_bottom_label)
            train_bottom_iou.update(torch.sigmoid(bottom_pred_source), source_bottom_label)

            # 更新进度条描述
            if wandb.config['adaptation_method'] in ['madann', 'dann']:
                train_bar.set_description(
                    f"Epoch {epoch}, Loss: {loss.item():.4f}, Adv Loss: {loss_adv.item():.4f}")
            else:
                train_bar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        # Reset metrics
        train_seg_acc.reset()
        train_seg_iou.reset()
        train_inst_acc.reset()
        train_inst_f1.reset()
        train_bottom_acc.reset()
        train_bottom_iou.reset()

        # Validation and testing phases...
        with torch.no_grad():
            with ema.average_parameters():
                model.eval()
                val_losses = []

                for data in tqdm(val_loader, desc="Validation"):
                    if data is None:
                        print("Data is None")
                        continue

                    target_graphs = data["target_graph"].to(device)

                    target_inst_label = data["target_inst_labels"].to(device)

                    target_seg_label = target_graphs.ndata["seg_y"].to(device)

                    target_bottom_label = target_graphs.ndata["bottom_y"].to(device)


                    seg_pred_target, inst_pred_target, bottom_pred_target, _ = model(target_graphs)

                    loss_seg_target = combined_loss_fn(seg_pred_target,
                                                       target_seg_label) if target_seg_label.sum() > 0 else 0
                    loss_inst_target = nn.BCEWithLogitsLoss()(inst_pred_target,
                                                              target_inst_label.float()) if target_inst_label.sum() > 0 else 0
                    loss_bottom_target = nn.BCEWithLogitsLoss()(bottom_pred_target,
                                                                target_bottom_label.float()) if target_bottom_label.sum() > 0 else 0

                    loss = wandb.config['seg_a'] * loss_seg_target + \
                           wandb.config['inst_a'] * loss_inst_target + \
                           wandb.config['bottom_a'] * loss_bottom_target

                    val_losses.append(loss.item())

                    val_seg_acc.update(F.softmax(seg_pred_target, dim=-1), target_seg_label)
                    val_seg_iou.update(F.softmax(seg_pred_target, dim=-1), target_seg_label)
                    val_inst_acc.update(torch.sigmoid(inst_pred_target), target_inst_label)
                    val_inst_f1.update(torch.sigmoid(inst_pred_target), target_inst_label)
                    val_bottom_acc.update(torch.sigmoid(bottom_pred_target), target_bottom_label)
                    val_bottom_iou.update(torch.sigmoid(bottom_pred_target), target_bottom_label)

                mean_val_loss = np.mean(val_losses).item() if val_losses else float('nan')
                mean_val_seg_acc = val_seg_acc.compute().item()
                mean_val_seg_iou = val_seg_iou.compute().item()
                mean_val_inst_acc = val_inst_acc.compute().item()
                mean_val_inst_f1 = val_inst_f1.compute().item()
                mean_val_bottom_acc = val_bottom_acc.compute().item()
                mean_val_bottom_iou = val_bottom_iou.compute().item()

                logger.info(f'val_loss : {mean_val_loss}, '
                            f'val_seg_acc: {mean_val_seg_acc}, '
                            f'val_seg_iou: {mean_val_seg_iou}, '
                            f'val_inst_acc: {mean_val_inst_acc}, '
                            f'val_inst_f1: {mean_val_inst_f1}, '
                            f'val_bottom_acc: {mean_val_bottom_acc}, '
                            f'val_bottom_iou: {mean_val_bottom_iou}')
                wandb.log({'epoch': epoch,
                           'val_loss': mean_val_loss,
                           'val_seg_acc': mean_val_seg_acc,
                           'val_seg_iou': mean_val_seg_iou,
                           'val_inst_acc': mean_val_inst_acc,
                           'val_inst_f1': mean_val_inst_f1,
                           'val_bottom_acc': mean_val_bottom_acc,
                           'val_bottom_iou': mean_val_bottom_iou})

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
                    torch.save(model.state_dict(), os.path.join(save_path, f"weight_{epoch}-epoch.pth"))

    # Testing phase...
    test_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        split='test',
        center_and_scale=False,
        normalize=True,
        random_rotate=True,
        num_threads=8,
        nums_data=wandb.config['test_target_data_nums']
    )
    test_loader = test_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    # 初始化目标域的度量指标
    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_inst_acc = BinaryAccuracy().to(device)
    test_bottom_acc = BinaryAccuracy().to(device)
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    test_inst_f1 = BinaryF1Score().to(device)
    test_bottom_iou = BinaryJaccardIndex().to(device)

    print('Start testing on target domain')
    with torch.no_grad():
        logger.info(f'------------- Now start testing on target domain ------------- ')
        model.eval()
        test_losses = []

        for data in tqdm(test_loader, desc="Testing"):
            if data is None:
                print("Data is None")
                continue

            target_graphs = data["target_graph"].to(device)
            target_inst_label = data["target_inst_labels"].to(device)
            target_seg_label = target_graphs.ndata["seg_y"].to(device)
            target_bottom_label = target_graphs.ndata["bottom_y"].to(device)

            seg_pred_target, inst_pred_target, bottom_pred_target, _ = model(target_graphs)


            loss_seg_target = combined_loss_fn(seg_pred_target, target_seg_label) if target_seg_label.sum() > 0 else 0
            loss_inst_target = nn.BCEWithLogitsLoss()(inst_pred_target,
                                                      target_inst_label.float()) if target_inst_label.sum() > 0 else 0
            loss_bottom_target = nn.BCEWithLogitsLoss()(bottom_pred_target,
                                                        target_bottom_label.float()) if target_bottom_label.sum() > 0 else 0

            loss = wandb.config['seg_a'] * loss_seg_target + \
                   wandb.config['inst_a'] * loss_inst_target + \
                   wandb.config['bottom_a'] * loss_bottom_target

            test_losses.append(loss.item())

            # 更新目标域的度量指标
            test_seg_acc.update(F.softmax(seg_pred_target, dim=-1), target_seg_label)
            test_seg_iou.update(F.softmax(seg_pred_target, dim=-1), target_seg_label)
            test_inst_acc.update(torch.sigmoid(inst_pred_target), target_inst_label)
            test_inst_f1.update(torch.sigmoid(inst_pred_target), target_inst_label)
            test_bottom_acc.update(torch.sigmoid(bottom_pred_target), target_bottom_label)
            test_bottom_iou.update(torch.sigmoid(bottom_pred_target), target_bottom_label)

    # 计算目标域的平均损失和各项指标
    mean_test_loss = np.mean(test_losses).item()
    mean_test_seg_acc = test_seg_acc.compute().item()
    mean_test_seg_iou = test_seg_iou.compute().item()
    mean_test_inst_acc = test_inst_acc.compute().item()
    mean_test_inst_f1 = test_inst_f1.compute().item()
    mean_test_bottom_acc = test_bottom_acc.compute().item()
    mean_test_bottom_iou = test_bottom_iou.compute().item()

    logger.info(f'test_loss : {mean_test_loss}, '
                f'test_seg_acc: {mean_test_seg_acc}, '
                f'test_seg_iou: {mean_test_seg_iou}, '
                f'test_inst_acc: {mean_test_inst_acc}, '
                f'test_inst_f1: {mean_test_inst_f1}, '
                f'test_bottom_acc: {mean_test_bottom_acc}, '
                f'test_bottom_iou: {mean_test_bottom_iou}')
    wandb.log({'test_loss': mean_test_loss,
               'test_seg_acc': mean_test_seg_acc,
               'test_seg_iou': mean_test_seg_iou,
               'test_inst_acc': mean_test_inst_acc,
               'test_inst_f1': mean_test_inst_f1,
               'test_bottom_acc': mean_test_bottom_acc,
               'test_bottom_iou': mean_test_bottom_iou})