# loss.py

import torch
from torch import nn
from torch.autograd import Function, grad
import torch.nn.functional as F
# Entropy loss function
def entropy_loss(predict_prob, epsilon=1e-20):
    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.mean(entropy)

def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None, num_samples=256):
    """
    Sample-based Maximum Mean Discrepancy (MMD) loss with Gaussian kernel to reduce memory usage.

    Args:
        source_features (torch.Tensor): Source domain features, shape [N, D]
        target_features (torch.Tensor): Target domain features, shape [M, D]
        kernel_mul (float): Multiplier for Gaussian kernel bandwidth.
        kernel_num (int): Number of kernels.
        fix_sigma (float): Predefined bandwidth for the kernel. If None, bandwidth is automatically calculated.
        num_samples (int): Number of samples to use for computing MMD.

    Returns:
        torch.Tensor: Computed MMD loss.
    """
    N, D = source_features.shape
    M, _ = target_features.shape

    # 随机采样特征对
    idx_s = torch.randint(0, N, (num_samples,), device=source_features.device)
    idx_t = torch.randint(0, M, (num_samples,), device=target_features.device)

    sampled_source = source_features[idx_s]
    sampled_target = target_features[idx_t]

    # 拼接采样后的特征
    total = torch.cat([sampled_source, sampled_target], dim=0)  # [2*num_samples, D]

    # 计算 L2 距离
    L2_distance = torch.cdist(total, total, p=2) ** 2  # [2*num_samples, 2*num_samples]

    # 计算带宽
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (2 * num_samples * num_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 计算高斯核
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    kernels = sum(kernel_val)  # [2*num_samples, 2*num_samples]

    # 分割核矩阵
    XX = kernels[:num_samples, :num_samples]
    YY = kernels[num_samples:, num_samples:]
    XY = kernels[:num_samples, num_samples:]

    # 计算 MMD 损失
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd



# TCA loss function (简化版)
def tca_loss(source_features, target_features, dim=30):
    """
    Transfer Component Analysis (TCA) loss.
    """
    X = torch.cat([source_features, target_features], dim=0)
    X_mean = torch.mean(X, 0, keepdim=True)
    X = X - X_mean
    U, S, V = torch.svd(X)
    Z = torch.mm(X, V[:, :dim])
    Z_s = Z[:source_features.size(0)]
    Z_t = Z[source_features.size(0):]
    loss = torch.norm(Z_s.mean(0) - Z_t.mean(0))
    return loss


# Gradient reversal layer for DANN
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

# # Wasserstein loss function
# def wasserstein_loss(y_pred, y_true):
#     return torch.mean(y_true * y_pred - (1 - y_true) * y_pred)


# Gradient penalty with batch size alignment
def gradient_penalty(discriminator, real_data, fake_data, device='cpu'):
    # 取源域和目标域中较小的batch size
    min_size = min(real_data.size(0), fake_data.size(0))
    real_data = real_data[:min_size]  # 截断源域数据
    fake_data = fake_data[:min_size]  # 截断目标域数据

    alpha = torch.rand(min_size, 1).to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True).to(device)

    disc_interpolates = discriminator(interpolates)

    gradients = grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 定义Balanced Focal Loss
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if self.alpha is not None:
            self.alpha = self.alpha.to(dtype=torch.float32)

    def forward(self, inputs, targets):
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
        else:
            alpha_t = torch.ones_like(targets, dtype=torch.float32, device=targets.device)

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
    def __init__(self, num_classes):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = (inputs + targets_one_hot).sum(dim=(0, 2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou.mean()

# 合并的损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', num_classes=25):
        super(CombinedLoss, self).__init__()
        self.focal_loss = BalancedFocalLoss(alpha, gamma, reduction)
        self.iou_loss = IoULoss(num_classes)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        iou_loss = self.iou_loss(inputs, targets)
        return focal_loss + iou_loss



