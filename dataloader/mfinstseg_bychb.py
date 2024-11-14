# import pathlib
# import json
# import torch
# import dgl
# import numpy as np
# from torch.utils.data import DataLoader
# import threading
# from torch.utils.data import Sampler
#
# from dataloader.base import BaseDataset
# from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
#
#
# class DomainBatchSampler(Sampler):
#     def __init__(self, data_source_size, data_target_size, batch_size):
#         self.data_source_size = data_source_size
#         self.data_target_size = data_target_size
#         self.batch_size = batch_size
#         # 计算每个域内的批次大小（batch_size的一半）
#         self.half_batch_size = batch_size // 2
#
#         # 计算可生成的完整批次数量
#         self.num_batches = min(self.data_source_size // self.half_batch_size,
#                                self.data_target_size // self.half_batch_size)
#
#     def __iter__(self):
#         indices = []
#         for i in range(self.num_batches):
#             # 分别生成源域和目标域的批次索引
#             source_indices = list(range(i * self.half_batch_size, (i + 1) * self.half_batch_size))
#             target_indices = list(range(i * self.half_batch_size, (i + 1) * self.half_batch_size))
#             # 合并并打乱索引
#             batch_indices = source_indices + target_indices
#             indices.extend(batch_indices)
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_batches * self.batch_size
#
#
# class MFInstSegAdaptiveDataset(BaseDataset):
#     remap_dict = {1: 0, 12: 1, 14: 2, 6: 3, 0: 4, 23: 5, 24: 6}
#
#     # 用于缓存已经加载的源域和目标域数据
#     cached_graphs_source = None
#     cached_graphs_target = None
#
#     @staticmethod
#     def num_classes(type='full'):
#         return 25 if type == 'full' else 7
#
#     @staticmethod
#     def remap(logit, dataset_type):
#         # 仅当数据集类型为 'tiny' 时才进行映射
#         if dataset_type == 'tiny':
#             if logit in MFInstSegAdaptiveDataset.remap_dict:
#                 return MFInstSegAdaptiveDataset.remap_dict[logit]
#             else:
#                 # 返回一个无效类别，例如 -1
#                 return -1
#         else:
#             return logit
#
#     def __init__(self,
#                  source_dir,
#                  target_dir,
#                  graphs_source=None,
#                  graphs_target=None,
#                  source_dataset_type='full',
#                  target_dataset_type='tiny',
#                  split="train",
#                  normalize=True,
#                  center_and_scale=True,
#                  random_rotate=False,
#                  nums_data=None,
#                  transform=None,
#                  num_threads=0):
#         super().__init__(transform, random_rotate)
#         self.source_path = pathlib.Path(source_dir)
#         self.target_path = pathlib.Path(target_dir)
#         self.source_dataset_type = source_dataset_type
#         self.target_dataset_type = target_dataset_type
#         self.normalize = normalize
#         self.center_and_scale = center_and_scale
#         assert split in ("train", "val", "test", "all")
#
#         # 加载文件列表
#         self.source_file_paths = self._load_files(self.source_path, split, nums_data)
#         self.target_file_paths = self._load_files(self.target_path, split, nums_data)
#         print(
#             f"Loaded {len(self.source_file_paths)} source files and {len(self.target_file_paths)} target files for split {split}")
#
#         # 如果是训练阶段且缓存为空，则加载数据并缓存
#         if split == 'train':
#             if MFInstSegAdaptiveDataset.cached_graphs_source is None:
#                 print("Loading source graphs for training...")
#                 self.graphs_source = self.load_graphs(self.source_path, self.source_file_paths, True, num_threads,
#                                                       self.source_dataset_type)
#                 MFInstSegAdaptiveDataset.cached_graphs_source = self.graphs_source
#             else:
#                 print("Using cached source graphs for training...")
#                 self.graphs_source = MFInstSegAdaptiveDataset.cached_graphs_source
#
#             if MFInstSegAdaptiveDataset.cached_graphs_target is None:
#                 print("Loading target graphs for training...")
#                 self.graphs_target = self.load_graphs(self.target_path, self.target_file_paths, False, num_threads,
#                                                       self.target_dataset_type)
#                 MFInstSegAdaptiveDataset.cached_graphs_target = self.graphs_target
#             else:
#                 print("Using cached target graphs for training...")
#                 self.graphs_target = MFInstSegAdaptiveDataset.cached_graphs_target
#         else:
#             # 在验证或测试阶段复用缓存的数据
#             print(f"Using cached graphs for {split} phase...")
#             self.graphs_source = MFInstSegAdaptiveDataset.cached_graphs_source
#             self.graphs_target = MFInstSegAdaptiveDataset.cached_graphs_target
#
#         self.data_source = self.graphs_source  # 源域数据
#         self.data_target = self.graphs_target  # 目标域数据
#
#     def _load_files(self, base_path, split, nums_data=None):
#         if split == "all":
#             with open(base_path / 'train.txt', 'r') as f:
#                 train_filelist = [x.strip() for x in f.readlines()]
#             with open(base_path / 'val.txt', 'r') as f:
#                 valid_filelist = [x.strip() for x in f.readlines()]
#             with open(base_path / 'test.txt', 'r') as f:
#                 test_filelist = [x.strip() for x in f.readlines()]
#             split_filelist = train_filelist + valid_filelist + test_filelist
#         else:
#             with open(base_path / f'{split}.txt', 'r') as f:
#                 split_filelist = [x.strip() for x in f.readlines()]
#
#         # 截取文件列表，如果num_train_data指定了，则截取前num_train_data个文件
#         if nums_data is not None:
#             split_filelist = split_filelist[:nums_data]
#
#         return split_filelist
#
#     def load_graphs(self, file_path, split_file_list, has_labels, num_threads=4, dataset_type='full'):
#         data = []
#         # 加载图数据
#         dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
#
#         if self.normalize:
#             stat = load_statistics(file_path.joinpath("aag").joinpath('attr_stat.json'))
#         else:
#             stat = None
#
#         chunk_size = (len(dataset) + num_threads - 1) // num_threads
#         chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
#
#         threads = []
#         results = [[] for _ in range(num_threads)]
#         for i in range(num_threads):
#             t = threading.Thread(target=lambda i: results[i].extend(
#                 self.process_chunk(
#                     chunks[i], split_file_list,
#                     self.normalize, self.center_and_scale, stat, file_path, has_labels, dataset_type)), args=(i,))
#             threads.append(t)
#             t.start()
#
#         for t in threads:
#             t.join()
#
#         data = [item for sublist in results for item in sublist]
#         return data
#
#     def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path,
#                       has_labels, dataset_type):
#         result = []
#         for one_data in chunk:
#             fn, data = one_data
#             if fn in split_file_list:
#                 one_graph = self.load_one_graph(fn, data, has_labels, dataset_type)
#
#                 if one_graph is None:
#                     continue
#                 if one_graph["graph"].edata["x"].size(0) == 0:
#                     continue
#                 if normalization_attribute and stat:
#                     one_graph = standardization(one_graph, stat)
#                 if center_and_scale_grid:
#                     one_graph = center_and_scale(one_graph)
#                 result.append(one_graph)
#         return result
#
#     def load_one_graph(self, fn, data, has_labels, dataset_type):
#         sample = load_one_graph(fn, data)
#         if sample is None:
#             print(f"Failed to load graph for file: {fn}")
#             return None
#
#         num_faces = sample['graph'].num_nodes()
#         label_file = self.source_path.joinpath("labels").joinpath(
#             fn + ".json") if has_labels else self.target_path.joinpath("labels").joinpath(fn + ".json")
#
#         if not label_file.exists():
#             print(f"Label file not found: {label_file}")
#             return None
#
#         with open(str(label_file), "r") as read_file:
#             labels_data = json.load(read_file)
#         _, labels = labels_data[0]
#         seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']
#
#         assert len(seg_label) == len(inst_label) and len(seg_label) == len(bottom_label)
#         assert num_faces == len(seg_label)
#
#         face_segmentation_labels = np.zeros(num_faces)
#         for idx, face_id in enumerate(range(num_faces)):
#             index = seg_label[str(face_id)]
#             mapped_index = self.remap(index, dataset_type)
#             if mapped_index == -1:
#                 print(f"Warning: Found invalid label {index} in file {fn}, mapped to -1")
#             face_segmentation_labels[idx] = mapped_index
#
#         instance_label = np.array(inst_label, dtype=np.int32)
#         bottom_segmentation_labels = np.zeros(num_faces)
#         for idx, face_id in enumerate(range(num_faces)):
#             index = bottom_label[str(face_id)]
#             bottom_segmentation_labels[idx] = index
#
#         sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentation_labels).long()
#         sample["inst_y"] = torch.tensor(instance_label).float()
#         sample["graph"].ndata["bottom_y"] = torch.tensor(bottom_segmentation_labels).float().reshape(-1, 1)
#         sample['domain'] = 'source' if has_labels else 'target'
#         sample["graph"].ndata['domain'] = torch.tensor([1 if has_labels else 0] * num_faces, dtype=torch.long)
#         return sample
#
#     def __getitem__(self, idx):
#         # 确保索引在范围内
#         if idx >= len(self):
#             raise IndexError(
#                 f"Index {idx} is out of range for data_source or data_target with length {len(self.data_source)} and {len(self.data_target)}.")
#
#         # 计算源域和目标域的索引
#         source_idx = idx % len(self.data_source)
#         target_idx = idx % len(self.data_target)
#
#         source_data = self.data_source[source_idx]
#         target_data = self.data_target[target_idx]
#
#         return {"source_data": source_data, "target_data": target_data}
#
#     def __len__(self):
#         # 数据集的总长度为源域和目标域样本数的最小值
#         return min(len(self.data_source), len(self.data_target))
#
#     def collate(self, batch):
#         source_graphs = [item["source_data"]["graph"] for item in batch]
#         target_graphs = [item["target_data"]["graph"] for item in batch]
#
#         batched_source_graph = dgl.batch(source_graphs)
#         batched_target_graph = dgl.batch(target_graphs)
#
#         source_inst_labels = self.pack_pad_2D_adj([item["source_data"] for item in batch])
#         target_inst_labels = self.pack_pad_2D_adj([item["target_data"] for item in batch])
#
#         return {
#             "source_graph": batched_source_graph,
#             "target_graph": batched_target_graph,
#             "source_inst_labels": source_inst_labels,
#             "target_inst_labels": target_inst_labels
#         }
#
#     def pack_pad_2D_adj(self, batch):
#         max_nodes = max([sample["graph"].num_nodes() for sample in batch])
#         batch_size = len(batch)
#         inst_labels = torch.zeros((batch_size, max_nodes, max_nodes))
#         for i, sample in enumerate(batch):
#             num_nodes = sample["graph"].num_nodes()
#             inst_labels[i, :num_nodes, :num_nodes] = sample["inst_y"]
#         return inst_labels
#
#     def get_dataloader(self, batch_size=256, shuffle=False, num_workers=0, drop_last=True, pin_memory=False):
#         sampler = DomainBatchSampler(len(self.data_source), len(self.data_target), batch_size)
#         return DataLoader(
#             self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             collate_fn=self.collate,
#             num_workers=num_workers,
#             drop_last=drop_last,
#             pin_memory=pin_memory
#         )
#
#
# if __name__ == '__main__':
#     source_dir = '/mnt/data/CHB/AAGNet-main/MFInstSeg'
#     target_dir = '/mnt/data/CHB/AAGNet-main/dataset/data_3w'
#
#     # 指定源域和目标域的数据集类型
#     source_dataset_type = 'full'
#     target_dataset_type = 'tiny'
#
#     # 创建数据集实例
#     train_dataset = MFInstSegAdaptiveDataset(
#         source_dir=source_dir,
#         target_dir=target_dir,
#         source_dataset_type=source_dataset_type,
#         target_dataset_type=target_dataset_type,
#         split='train',
#         center_and_scale=True,
#         normalize=False,
#         num_threads=8,
#         nums_data=100
#     )
#
#     # 打印数据集大小
#     print(f"Total dataset size: {len(train_dataset)}")
#     source_data_size = len(train_dataset.data_source)
#     target_data_size = len(train_dataset.data_target)
#     print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")
#
#     # 创建数据加载器
#     train_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)
#
#     # 创建数据集实例
#     test_dataset = MFInstSegAdaptiveDataset(
#         source_dir=source_dir,
#         target_dir=target_dir,
#         source_dataset_type=source_dataset_type,
#         target_dataset_type=target_dataset_type,
#         split='test',
#         center_and_scale=True,
#         normalize=False,
#         num_threads=8,
#         nums_data=100
#     )
#     test_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)
#     # 遍历一个批次数据
#     for data in test_loader:
#         if data is not None:
#             print("Source domain graph:")
#             print(data["source_graph"])  # 打印源域图数据
#             print("Target domain graph:")
#             print(data["target_graph"])  # 打印目标域图数据
#
#             # 打印源域和目标域的实例标签
#             print("Source instance labels:")
#             print(data["source_inst_labels"])
#             print("Target instance labels:")
#             print(data["target_inst_labels"])
#
#             # 打印源域和目标域的语义分割标签
#             print("Source segmentation labels:")
#             print(data["source_graph"].ndata["seg_y"])
#             print("Target segmentation labels:")
#             print(data["target_graph"].ndata["seg_y"])
#
#             break  # 只打印第一个批次的数据，之后跳出循环


import pathlib
import json
import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader
import threading
from torch.utils.data import Sampler

from dataloader.base import BaseDataset
from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale


class DomainBatchSampler(Sampler):
    def __init__(self, data_source_size, data_target_size, batch_size):
        self.data_source_size = data_source_size
        self.data_target_size = data_target_size
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2

        # 计算源域和目标域的完整批次数量
        self.num_batches_source = data_source_size // self.half_batch_size
        self.num_batches_target = data_target_size // self.half_batch_size
        self.num_batches = min(self.num_batches_source, self.num_batches_target)

    def __iter__(self):
        indices = []
        for i in range(self.num_batches):
            # 分别生成源域和目标域的批次索引
            source_indices = list(range(i * self.half_batch_size, (i + 1) * self.half_batch_size))
            target_indices = list(range(i * self.half_batch_size, (i + 1) * self.half_batch_size))

            # 如果索引超过数据集大小则使用模运算符
            source_indices = [idx % self.data_source_size for idx in source_indices]
            target_indices = [idx % self.data_target_size for idx in target_indices]

            # 合并并打乱索引
            batch_indices = source_indices + target_indices
            indices.extend(batch_indices)

        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size


# class DomainBatchSampler(Sampler):
#     def __init__(self, data_source_size, data_target_size, batch_size):
#         self.data_source_size = data_source_size
#         self.data_target_size = data_target_size
#         self.batch_size = batch_size
#         # 计算每个域内的批次大小（batch_size的一半）
#         self.half_batch_size = batch_size // 2
#
#         # 计算可生成的完整批次数量
#         self.num_batches = min(self.data_source_size // self.half_batch_size,
#                                self.data_target_size // self.half_batch_size)
#
#     def __iter__(self):
#         indices = []
#         for i in range(self.num_batches):
#             # 分别生成源域和目标域的批次索引
#             source_indices = np.random.choice(self.data_source_size, self.half_batch_size, replace=False)
#             target_indices = np.random.choice(self.data_target_size, self.half_batch_size, replace=False)
#             # 合并并打乱索引
#             batch_indices = np.concatenate([source_indices, target_indices])
#             np.random.shuffle(batch_indices)
#             indices.extend(batch_indices)
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_batches * self.batch_size


class MFInstSegAdaptiveDataset(BaseDataset):
    remap_dict = {1: 0, 12: 1, 14: 2, 6: 3, 0: 4, 23: 5, 24: 6}

    # 用于缓存已经加载的源域和目标域数据
    cached_graphs_source = None
    cached_graphs_target = None

    @staticmethod
    def num_classes(type='full'):
        return 25 if type == 'full' else 7

    @staticmethod
    def remap(logit, dataset_type):
        # 仅当数据集类型为 'tiny' 时才进行映射
        if dataset_type == 'tiny':
            if logit in MFInstSegAdaptiveDataset.remap_dict:
                return MFInstSegAdaptiveDataset.remap_dict[logit]
            else:
                # 返回一个无效类别，例如 -1
                return -1
        else:
            return logit

    def __init__(self,
                 source_dir,
                 target_dir,
                 graphs_source=None,
                 graphs_target=None,
                 source_dataset_type='full',
                 target_dataset_type='tiny',
                 split="train",
                 normalize=True,
                 center_and_scale=True,
                 random_rotate=False,
                 nums_data=None,
                 transform=None,
                 num_threads=0):
        super().__init__(transform, random_rotate)
        self.source_path = pathlib.Path(source_dir)
        self.target_path = pathlib.Path(target_dir)
        self.source_dataset_type = source_dataset_type
        self.target_dataset_type = target_dataset_type
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        assert split in ("train", "val", "test", "all")

        # 加载文件列表
        self.source_file_paths = self._load_files(self.source_path, split, nums_data)
        self.target_file_paths = self._load_files(self.target_path, split, nums_data)
        print(
            f"Loaded {len(self.source_file_paths)} source files and {len(self.target_file_paths)} target files for split {split}")

        if split == 'train':
            # 训练阶段的逻辑，与之前相同
            if MFInstSegAdaptiveDataset.cached_graphs_source is None:
                print("Loading source graphs for training...")
                self.graphs_source = self.load_graphs(self.source_path, self.source_file_paths, True, num_threads,
                                                      self.source_dataset_type)
                MFInstSegAdaptiveDataset.cached_graphs_source = self.graphs_source
            else:
                print("Using cached source graphs for training...")
                self.graphs_source = MFInstSegAdaptiveDataset.cached_graphs_source

            if MFInstSegAdaptiveDataset.cached_graphs_target is None:
                print("Loading target graphs for training...")
                self.graphs_target = self.load_graphs(self.target_path, self.target_file_paths, False, num_threads,
                                                      self.target_dataset_type)
                MFInstSegAdaptiveDataset.cached_graphs_target = self.graphs_target
            else:
                print("Using cached target graphs for training...")
                self.graphs_target = MFInstSegAdaptiveDataset.cached_graphs_target
        else:
            # 在验证或测试阶段，首先检查缓存是否存在
            if MFInstSegAdaptiveDataset.cached_graphs_source is None:
                print("Loading source graphs for {}...".format(split))
                self.graphs_source = self.load_graphs(self.source_path, self.source_file_paths, True, num_threads,
                                                      self.source_dataset_type)
            else:
                print("Using cached source graphs for {}...".format(split))
                self.graphs_source = MFInstSegAdaptiveDataset.cached_graphs_source

            if MFInstSegAdaptiveDataset.cached_graphs_target is None:
                print("Loading target graphs for {}...".format(split))
                self.graphs_target = self.load_graphs(self.target_path, self.target_file_paths, False, num_threads,
                                                      self.target_dataset_type)
            else:
                print("Using cached target graphs for {}...".format(split))
                self.graphs_target = MFInstSegAdaptiveDataset.cached_graphs_target

        self.data_source = self.graphs_source  # 源域数据
        self.data_target = self.graphs_target  # 目标域数据
    def _load_files(self, base_path, split, nums_data=None):
        if split == "all":
            with open(base_path / 'train.txt', 'r') as f:
                train_filelist = [x.strip() for x in f.readlines()]
            with open(base_path / 'val.txt', 'r') as f:
                valid_filelist = [x.strip() for x in f.readlines()]
            with open(base_path / 'test.txt', 'r') as f:
                test_filelist = [x.strip() for x in f.readlines()]
            split_filelist = train_filelist + valid_filelist + test_filelist
        else:
            with open(base_path / f'{split}.txt', 'r') as f:
                split_filelist = [x.strip() for x in f.readlines()]

        # 截取文件列表，如果num_train_data指定了，则截取前num_train_data个文件
        if nums_data is not None:
            split_filelist = split_filelist[:nums_data]

        return split_filelist

    def load_graphs(self, file_path, split_file_list, has_labels, num_threads=4, dataset_type='full'):
        data = []
        dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))

        if self.normalize:
            stat = load_statistics(file_path.joinpath("aag").joinpath('attr_stat.json'))
        else:
            stat = None

        chunk_size = (len(dataset) + num_threads - 1) // num_threads
        chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

        threads = []
        results = [[] for _ in range(num_threads)]
        for i in range(num_threads):
            t = threading.Thread(target=lambda i: results[i].extend(
                self.process_chunk(
                    chunks[i], split_file_list,
                    self.normalize, self.center_and_scale, stat, file_path, has_labels, dataset_type)), args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 确保数据顺序不被打乱
        data = sorted([item for sublist in results for item in sublist], key=lambda x: x['filename'])
        return data

    def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path,
                      has_labels, dataset_type):
        result = []
        for one_data in chunk:
            fn, data = one_data
            if fn in split_file_list:
                one_graph = self.load_one_graph(fn, data, has_labels, dataset_type)

                if one_graph is None:
                    continue
                if one_graph["graph"].edata["x"].size(0) == 0:
                    continue
                if normalization_attribute and stat:
                    one_graph = standardization(one_graph, stat)
                if center_and_scale_grid:
                    one_graph = center_and_scale(one_graph)
                result.append(one_graph)
        return result

    def load_one_graph(self, fn, data, has_labels, dataset_type):
        sample = load_one_graph(fn, data)
        if sample is None:
            print(f"Failed to load graph for file: {fn}")
            return None

        num_faces = sample['graph'].num_nodes()
        label_file = self.source_path.joinpath("labels").joinpath(
            fn + ".json") if has_labels else self.target_path.joinpath("labels").joinpath(fn + ".json")

        if not label_file.exists():
            print(f"Label file not found: {label_file}")
            return None

        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        _, labels = labels_data[0]
        seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']

        assert len(seg_label) == len(inst_label) and len(seg_label) == len(bottom_label)
        assert num_faces == len(seg_label)

        face_segmentation_labels = np.zeros(num_faces)
        for idx, face_id in enumerate(range(num_faces)):
            index = seg_label[str(face_id)]
            mapped_index = self.remap(index, dataset_type)
            if mapped_index == -1:
                print(f"Warning: Found invalid label {index} in file {fn}, mapped to -1")
            face_segmentation_labels[idx] = mapped_index

        instance_label = np.array(inst_label, dtype=np.int32)
        bottom_segmentation_labels = np.zeros(num_faces)
        for idx, face_id in enumerate(range(num_faces)):
            index = bottom_label[str(face_id)]
            bottom_segmentation_labels[idx] = index

        sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentation_labels).long()
        sample["inst_y"] = torch.tensor(instance_label).float()
        sample["graph"].ndata["bottom_y"] = torch.tensor(bottom_segmentation_labels).float().reshape(-1, 1)
        sample['domain'] = 'source' if has_labels else 'target'
        sample["graph"].ndata['domain'] = torch.tensor([1 if has_labels else 0] * num_faces, dtype=torch.long)


        return sample

    def __getitem__(self, idx):
        # 确保索引在范围内
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of range for data_source or data_target with length {len(self.data_source)} and {len(self.data_target)}.")

        # 使用sampler中的索引直接访问数据
        source_idx = idx % len(self.data_source)
        target_idx = idx % len(self.data_target)

        source_data = self.data_source[source_idx]
        target_data = self.data_target[target_idx]

        return {"source_data": source_data, "target_data": target_data}



    def __len__(self):
        # 数据集的总长度为源域和目标域样本数的最小值
        return min(len(self.data_source), len(self.data_target))

    def collate(self, batch):
        source_graphs = [item["source_data"]["graph"] for item in batch]
        target_graphs = [item["target_data"]["graph"] for item in batch]

        batched_source_graph = dgl.batch(source_graphs)
        batched_target_graph = dgl.batch(target_graphs)

        source_inst_labels = self.pack_pad_2D_adj([item["source_data"] for item in batch])
        target_inst_labels = self.pack_pad_2D_adj([item["target_data"] for item in batch])

        return {
            "source_graph": batched_source_graph,
            "target_graph": batched_target_graph,
            "source_inst_labels": source_inst_labels,
            "target_inst_labels": target_inst_labels
        }

    def pack_pad_2D_adj(self, batch):
        max_nodes = max([sample["graph"].num_nodes() for sample in batch])
        batch_size = len(batch)
        inst_labels = torch.zeros((batch_size, max_nodes, max_nodes))
        for i, sample in enumerate(batch):
            num_nodes = sample["graph"].num_nodes()
            inst_labels[i, :num_nodes, :num_nodes] = sample["inst_y"]
        return inst_labels

    def get_dataloader(self, batch_size=256, shuffle=False, num_workers=0, drop_last=True, pin_memory=False):
        sampler = DomainBatchSampler(len(self.data_source), len(self.data_target), batch_size)
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )
