import os
import time
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex
)
import wandb


from dataloader.Adapt_dataloader_MFTRCAD_MFInst import MFInstSegAdaptiveDataset
# from dataloader.Adapt_dataloader_MFCAD2_MFInst import MFInstSegAdaptiveDataset
# from dataloader.mfinstseg_bychb_v2 import MFInstSegAdaptiveDataset
# from dataloader.mfinstseg import MFInstSegDataset
from models.inst_segmentors_byChb_adapt_SegTask import SFRGNNSegmentor
from models.loss import entropy_loss, mmd_loss, tca_loss
from utils.misc import seed_torch, init_logger, print_num_params
import torch.nn.functional as F
from collections import Counter


def compute_class_weights(data_source, num_classes, device):
    class_counts = Counter()

    for idx, sample in enumerate(tqdm(data_source, desc="Computing class weights")):
        graph = sample["graph"]
        if "seg_y" not in graph.ndata:
            print(f"Warning: 'seg_y' key not found in graph.ndata for sample index {idx}. Skipping.")
            continue

        seg_labels = graph.ndata["seg_y"].cpu().numpy()

        # 确保标签在 [0, num_classes-1] 范围内
        seg_labels = np.clip(seg_labels, 0, num_classes - 1)

        class_counts.update(seg_labels.tolist())

    total_count = sum(class_counts.values())
    if total_count == 0:
        raise ValueError("No samples found in source domain for computing class weights.")

    # 计算每个类别的权重
    class_weights = {cls: total_count / count for cls, count in class_counts.items() if count > 0}

    # 归一化权重，使最大权重为1
    max_weight = max(class_weights.values())
    normalized_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    # 创建权重张量
    weights = torch.zeros(num_classes, device=device)
    for cls, weight in normalized_weights.items():
        if cls < num_classes:
            weights[cls] = weight
        else:
            print(
                f"Warning: Class index {cls} exceeds num_classes {num_classes}. Assigning to class {num_classes - 1}.")
            weights[num_classes - 1] += weight  # 将超出范围的权重累加到最后一个类别

    return weights


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
    torch.set_float32_matmul_precision("high")
    os.environ["WANDB_API_KEY"] = 'your_wandb_api_key'  # 替换为您的 WandB API Key
    os.environ["WANDB_MODE"] = "offline"

    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
        project="domain_adapt_methods_madann_MFTRCAD-MFInstSeg",
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
            "dataset_source": "/path/source_dataset",  # 注意路径分隔符
            "dataset_target": "/path/target_dataset",
            "source_dataset_type": "MFTRCAD",
            "target_dataset_type": "MFInstSeg",
            "train_data_nums": 16000,
            "val_data_nums": 3400,
            "test_target_data_nums": 3400,
            "epochs": 100,
            "lr": 0.001,
            "weight_decay": 1e-2,
            "batch_size": 256,
            "ema_decay_per_epoch": 1. / 2.                                                                                                                                                                                                                                                                                                                              ,
            "seg_a": 1,#1
            "adaptation_method": "madann",  # dann, mmd, tca
            "mmd_weight": 0.1,
            "tca_weight": 0.1,
            "wdann_weight": 0.001,
            "dann_weight": 0.008,#0.01 0.008
            "lambda_gp": 10,# 梯度惩罚的权重
            "lr_g": 0.01,  # 特征提取器的学习率 0.001 0.01
            "lr_d": 0.01,  # 域判别器的学习率 0.0001 0.01
            "d_steps": 1,  # 域判别器的训练步数
        }
    )

    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    dataset_type = wandb.config['dataset_type']
    n_classes = 25  # 0-24

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
        adaptation_method=wandb.config['adaptation_method']
    ).to(device)

    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params

    source_dir = wandb.config['dataset_source']
    target_dir = wandb.config['dataset_target']
    source_dataset_type = wandb.config['source_dataset_type']
    target_dataset_type = wandb.config['target_dataset_type']
    # 初始化数据集
    train_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,target_dir=target_dir,
        source_dataset_type=source_dataset_type,target_dataset_type=target_dataset_type,
        split='train',
        center_and_scale=False, normalize=True,
        num_threads=8,nums_data=wandb.config['train_data_nums'])

    print(f"Total train dataset size: {len(train_dataset)}")
    print(f"Source data size: {len(train_dataset.data_source)}, Target data size: {len(train_dataset.data_target)}")

    train_loader = train_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    val_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,target_dir=target_dir,
        source_dataset_type=source_dataset_type,target_dataset_type=target_dataset_type,
        split='val',
        center_and_scale=False,normalize=True,
        num_threads=8,nums_data=wandb.config['val_data_nums']
    )

    print(f"Validation dataset size: {len(val_dataset)}")

    val_loader = val_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    # 计算类别权重
    class_weights = compute_class_weights(train_dataset.data_source, num_classes=n_classes, device=device)
    print(f"Class weights: {class_weights}")

    # 定义损失函数
    combined_loss_fn = CombinedLoss(alpha=class_weights)
    # Metrics for segmentation
    train_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    train_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    val_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    val_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    # 定义优化器和调度器
    if wandb.config['adaptation_method'] in ['wdann', 'dann']:
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
        # 如果没有使用DANN或WDANN，使用一个全局优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'],
                                      weight_decay=wandb.config['weight_decay'])  # L2正则化

    # 学习率调度器配置
    if wandb.config['adaptation_method'] in ['wdann', 'dann']:
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
        logger.info(f'------------- Now start epoch {epoch}------------- ')
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        loss_adv = None

        loss_t = None
        for data in train_bar:
            if data is None:
                continue

            # Transfer data to device
            source_graphs = data["source_graph"].to(device, non_blocking=True)
            target_graphs = data["target_graph"].to(device, non_blocking=True)
            source_seg_label = source_graphs.ndata["seg_y"].to(device)
            target_seg_label = target_graphs.ndata["seg_y"].to(device)

            if wandb.config['adaptation_method'] == 'madann' or wandb.config['adaptation_method'] == 'dann':

                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                seg_pred_source, node_emb_source = model(source_graphs)
                seg_pred_target, node_emb_target = model(target_graphs)

                loss_seg_source = combined_loss_fn(seg_pred_source, source_seg_label)

                features = torch.cat((node_emb_source, node_emb_target), dim=0)
                domain_labels = torch.cat((
                    torch.zeros(node_emb_source.size(0)).to(device),
                    torch.ones(node_emb_target.size(0)).to(device)
                ), dim=0)

                from models.loss import grad_reverse

                features = grad_reverse(features)
                domain_output = model.domain_discriminator(features).squeeze()
                domain_loss = nn.BCEWithLogitsLoss()(domain_output, domain_labels)

                # # Entropy loss for target domain
                seg_prob_t = F.softmax(seg_pred_target, dim=1)
                loss_t = entropy_loss(seg_prob_t)

                loss = (wandb.config['seg_a'] * loss_seg_source +
                        wandb.config['dann_weight'] * domain_loss +
                        0.1 * loss_t)
                # loss = (wandb.config['seg_a'] * loss_seg_source +
                #         wandb.config['dann_weight'] * domain_loss)
                loss_adv = domain_loss  #

                loss.backward()
                optimizer_g.step()
                optimizer_d.step()
                # 监控域判别器输出
                wandb.log({
                    'domain_discriminator_output_s': torch.mean(model.domain_discriminator(node_emb_source)).item(),
                    'domain_discriminator_output_t': torch.mean(model.domain_discriminator(node_emb_target)).item()
                })

            elif wandb.config['adaptation_method'] == 'mmd':
                # ============== MMD ==============
                optimizer.zero_grad()

                seg_pred_source, node_emb_source = model(source_graphs)
                seg_pred_target, node_emb_target = model(target_graphs)

                loss_seg_source = combined_loss_fn(seg_pred_source, source_seg_label)

                loss_mmd = mmd_loss(node_emb_source, node_emb_target)

                # Entropy loss for target domain
                seg_prob_t = F.softmax(seg_pred_target, dim=1)
                loss_t = entropy_loss(seg_prob_t)

                loss = (wandb.config['seg_a'] * loss_seg_source +
                        wandb.config['mmd_weight'] * loss_mmd +
                        0.1 * loss_t)

                loss.backward()
                optimizer.step()

            elif wandb.config['adaptation_method'] == 'tca':
                # ============== TCA ==============
                optimizer.zero_grad()

                seg_pred_source, node_emb_source = model(source_graphs)
                seg_pred_target, node_emb_target = model(target_graphs)

                loss_seg_source = combined_loss_fn(seg_pred_source, source_seg_label)

                loss_tca = tca_loss(node_emb_source, node_emb_target)

                # Entropy loss for target domain
                seg_prob_t = F.softmax(seg_pred_target, dim=1)
                loss_t = entropy_loss(seg_prob_t)

                loss = (wandb.config['seg_a'] * loss_seg_source +
                        wandb.config['tca_weight'] * loss_tca +
                        0.1 * loss_t)

                loss.backward()
                optimizer.step()


            else:
                raise ValueError(f"Unknown adaptation method: {wandb.config['adaptation_method']}")

            ema.update()

            # Update metrics
            train_losses.append(loss.item())
            train_seg_acc.update(F.softmax(seg_pred_source, dim=1), source_seg_label)
            train_seg_iou.update(F.softmax(seg_pred_source, dim=1), source_seg_label)

            # Update progress bar description
            if wandb.config['adaptation_method'] in ['wdann', 'dann']:
                adv_loss_val = loss_adv.item() if loss_adv else 0.0
                train_bar.set_description(
                    f"Epoch:{epoch} Loss:{loss.item():.4f}, Adv Loss:{adv_loss_val:.4f}")

            else:
                train_bar.set_description(f"Epoch:{epoch} Loss:{loss.item():.4f}")

        # Learning rate scheduler step
        if wandb.config['adaptation_method'] in ['madann', 'dann']:
            scheduler_g.step()
            scheduler_d.step()
        else:
            scheduler.step()

        mean_train_loss = np.mean(train_losses)
        mean_train_seg_acc = train_seg_acc.compute().item()
        mean_train_seg_iou = train_seg_iou.compute().item()

        logger.info(
            f'train_loss : {mean_train_loss}, train_seg_acc: {mean_train_seg_acc}, train_seg_iou: {mean_train_seg_iou}')

        wandb_log = {
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'train_seg_acc': mean_train_seg_acc,
            'train_seg_iou': mean_train_seg_iou
        }
        if wandb.config['adaptation_method'] in ['madann', 'dann']:
            wandb_log['train_loss_adv'] = loss_adv.item() if loss_adv else 0.0
        wandb.log(wandb_log)

        train_seg_acc.reset()
        train_seg_iou.reset()

        # Validation phase
        with torch.no_grad():
            with ema.average_parameters():
                model.eval()
                val_losses = []

                for idx, data in enumerate(tqdm(val_loader, desc="Validation")):
                    if data is None:
                        print("Data is None")
                        continue

                    target_graphs = data["target_graph"].to(device)
                    target_seg_label = target_graphs.ndata["seg_y"].to(device)

                    # 只对目标域数据进行验证
                    seg_pred_target, _ = model(target_graphs)

                    # 计算目标域损失
                    loss_seg_target = combined_loss_fn(seg_pred_target, target_seg_label)

                    val_losses.append(loss_seg_target.item())

                    val_seg_acc.update(F.softmax(seg_pred_target, dim=1), target_seg_label)
                    val_seg_iou.update(F.softmax(seg_pred_target, dim=1), target_seg_label)

                mean_val_loss = np.mean(val_losses).item() if val_losses else float('nan')
                mean_val_seg_acc = val_seg_acc.compute().item()
                mean_val_seg_iou = val_seg_iou.compute().item()

                logger.info(
                    f'val_loss : {mean_val_loss}, val_seg_acc: {mean_val_seg_acc}, val_seg_iou: {mean_val_seg_iou}')
                wandb.log({
                    'epoch': epoch,
                    'val_loss': mean_val_loss,
                    'val_seg_acc': mean_val_seg_acc,
                    'val_seg_iou': mean_val_seg_iou
                })

                val_seg_acc.reset()
                val_seg_iou.reset()

                # 保存最佳模型
                if mean_val_seg_iou > best_acc:
                    best_acc = mean_val_seg_iou
                    logger.info(f'Best metric: {best_acc}, model saved')
                    torch.save(model.state_dict(), os.path.join(save_path, f"weight_{epoch}.pth"))

    # Testing phase
    test_dataset = MFInstSegAdaptiveDataset(source_dir=source_dir, target_dir=target_dir,
                                            source_dataset_type=source_dataset_type,
                                            target_dataset_type=target_dataset_type,
                                            split='test',
                                            center_and_scale=False, normalize=True,
                                            num_threads=8, nums_data=wandb.config['test_target_data_nums']
                                            )
    print(f"Test dataset size: {len(test_dataset)}")

    test_loader = test_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    print('Start testing on target domain')

    with torch.no_grad():
        logger.info(f'------------- Now start testing on target domain ------------- ')
        model.eval()
        test_losses = []

        for idx, data in enumerate(tqdm(test_loader, desc="Testing")):
            if data is None:
                print("Data is None")
                continue

            target_graphs = data["target_graph"].to(device)
            target_seg_label = target_graphs.ndata["seg_y"].to(device)

            seg_pred_target, _ = model(target_graphs)

            loss_seg_target = combined_loss_fn(seg_pred_target, target_seg_label)
            test_losses.append(loss_seg_target.item())

            test_seg_acc.update(F.softmax(seg_pred_target, dim=1), target_seg_label)
            test_seg_iou.update(F.softmax(seg_pred_target, dim=1), target_seg_label)

    mean_test_loss = np.mean(test_losses).item()
    mean_test_seg_acc = test_seg_acc.compute().item()
    mean_test_seg_iou = test_seg_iou.compute().item()

    logger.info(f'test_loss : {mean_test_loss}, test_seg_acc: {mean_test_seg_acc}, test_seg_iou: {mean_test_seg_iou}')
    wandb.log({
        'test_loss': mean_test_loss,
        'test_seg_acc': mean_test_seg_acc,
        'test_seg_iou': mean_test_seg_iou
    })