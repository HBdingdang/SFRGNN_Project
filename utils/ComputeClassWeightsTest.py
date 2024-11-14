import torch
from collections import Counter

from dataloader.mfinstseg import MFInstSegDataset


# # 假设这是你的数据集结构
# class MockDataset:
#     def __init__(self):
#         self.data = [
#             {"graph": {"ndata": {"seg_y": torch.tensor([0, 1, 1, 2, 2, 2])}}},
#             {"graph": {"ndata": {"seg_y": torch.tensor([0, 0, 1, 2, 2, 1])}}},
#             {"graph": {"ndata": {"seg_y": torch.tensor([1, 1, 1, 2, 0, 0])}}}
#         ]
#
#     def __iter__(self):
#         return iter(self.data)


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


# 创建一个模拟数据集
# mock_dataset = MockDataset()
test_dataset = MFInstSegDataset(root_dir='/mnt/data', split='train',
                                    center_and_scale=False, normalize=True, random_rotate=False,
                                    num_threads=8)
num_classes = 25  # 假设有3个类别

# 计算类别权重
class_weights = compute_class_weights(test_dataset, num_classes)

# 打印类别权重
print("Computed class weights:", class_weights)
