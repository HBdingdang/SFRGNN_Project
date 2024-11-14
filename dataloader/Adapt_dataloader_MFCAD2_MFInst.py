#MFInst->MFCAD2
import pathlib
import json
import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader, Sampler
from dataloader.base import BaseDataset
from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
import threading

# 自定义批采样器，用于确保每个批次有相同数量的源域和目标域样本
class DomainBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.source_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'source']
        self.target_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'target']
        self.num_batches = min(len(self.source_indices) // (batch_size // 2),
                               len(self.target_indices) // (batch_size // 2))

    def __iter__(self):
        indices = []
        for i in range(self.num_batches):
            source_batch = self.source_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
            target_batch = self.target_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
            indices.extend(source_batch + target_batch)
        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size


class MFInstSegAdaptiveDataset(BaseDataset):
    @staticmethod
    def num_classes(type='full'):
        return 25 if type == 'full' else 7

    def __init__(self,
                 source_dir,  # MFInstSeg 数据集
                 target_dir,  # MFCAD2 数据集
                 graphs_source=None,
                 graphs_target=None,
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
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        assert split in ("train", "val", "test", "all")

        # 加载源域 (MFInstSeg) 和目标域 (MFCAD2) 的数据集
        split_filelist_source = self._load_split_file(self.source_path, split, nums_data)
        split_filelist_target = self._load_split_file(self.target_path, split, nums_data)

        print(f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split {split}")

        self.graphs_source = self.load_graphs(self.source_path, graphs_source, split_filelist_source, True, num_threads, dataset_type='MFInstSeg')
        self.graphs_target = self.load_graphs(self.target_path, graphs_target, split_filelist_target, False, num_threads, dataset_type='MFCAD2')

        self.data_source = self.graphs_source
        self.data_target = self.graphs_target
        self.data = self.data_source + self.data_target
        print(f"Done loading {len(self.data)} files for {split} split")

    def _load_split_file(self, base_path, split, nums_data=None):
        """
        根据指定的 split 加载 MFInstSeg 或 MFCAD2 数据集的文件列表
        """
        split_txt_path = base_path / f'{split}.txt'
        if not split_txt_path.exists():
            raise FileNotFoundError(f"Could not find split file: {split_txt_path}")
        with open(split_txt_path, 'r') as f:
            split_filelist = [x.strip() for x in f.readlines()]

        if nums_data is not None:
            split_filelist = split_filelist[:nums_data]

        return split_filelist

    def load_graphs(self, file_path, graphs, split_file_list, has_labels, num_threads=4, dataset_type='MFInstSeg'):
        self.data = []
        if has_labels:
            if graphs:
                self.dataset = graphs
            else:
                self.dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
        else:
            if graphs:
                self.dataset = graphs
            else:
                self.dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))

        if self.normalize:
            stat = load_statistics(file_path.joinpath("aag").joinpath('attr_stat.json'))
        else:
            stat = None

        chunk_size = (len(self.dataset) + num_threads - 1) // num_threads
        chunks = [self.dataset[i:i + chunk_size] for i in range(0, len(self.dataset), chunk_size)]

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

        self.data = [item for sublist in results for item in sublist]
        return self.data

    def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path, has_labels, dataset_type):
        result = []
        for one_data in chunk:
            fn, data = one_data
            if fn in split_file_list:
                one_graph = self.load_one_graph(fn, data, dataset_type)
                if one_graph is None:
                    continue
                if normalization_attribute and stat:
                    one_graph = standardization(one_graph, stat)
                if center_and_scale_grid:
                    one_graph = center_and_scale(one_graph)
                result.append(one_graph)
        return result

    def load_one_graph(self, fn, data, dataset_type):
        """
        加载单个图并附加语义分割标签，区分 MFInstSeg 和 MFCAD2 数据集的不同标签结构。
        """
        sample = load_one_graph(fn, data)

        # 处理 MFInstSeg 源域数据
        if dataset_type == 'MFInstSeg':
            label_file = self.source_path.joinpath("labels").joinpath(fn + ".json")
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)

            # 从嵌套结构中提取 'seg' 标签数据
            seg_labels = labels_data[0][1]['seg']

            # 提取每个面的语义分割标签，按面ID排序
            labels_data = [seg_labels[str(face_id)] for face_id in range(sample['graph'].num_nodes())]

            # 将标签转换为张量
            labels_data = np.array(labels_data, dtype=np.int32)
            sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()

        # 处理 MFCAD2 目标域数据
        elif dataset_type == 'MFCAD2':
            label_file = self.target_path.joinpath("labels").joinpath(fn + ".json")
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)

            # 检查标签是否是字典还是列表/数组类型，并处理
            if isinstance(labels_data, dict):
                # 如果是字典，假设标签直接存储为一个键
                labels_data = labels_data.get('seg', labels_data)  # 如果是字典可能有 'seg' 键
            # 如果标签是一个整数列表或数组，直接使用
            elif isinstance(labels_data, list):
                pass  # 已是所需的格式

            # 将标签转换为张量
            labels_data = np.array(labels_data, dtype=np.int32)
            sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()

        # 域区分
        sample['domain'] = 'source' if dataset_type == 'MFInstSeg' else 'target'
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        source_graphs = []
        target_graphs = []

        # 将源域和目标域的图分别放入不同的列表
        for sample in batch:
            if sample['domain'] == 'source':
                source_graphs.append(sample['graph'])
            else:
                target_graphs.append(sample['graph'])

        # 分别对源域和目标域的图进行批处理
        batched_source_graph = dgl.batch(source_graphs)
        batched_target_graph = dgl.batch(target_graphs)

        return {"source_graph": batched_source_graph,
                "target_graph": batched_target_graph}

    def get_dataloader(self, batch_size=256, shuffle=False, sampler=None, num_workers=0, drop_last=True, pin_memory=False):
        sampler = DomainBatchSampler(self.data, batch_size)
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

if __name__ == '__main__':
    source_dir = 'E:\CHB\python_project\SFRGNN\MFInstSeg'  # MFInstSeg 数据集
    target_dir = 'E:\CHB\python_project\SFRGNN\MFCAD2'  # MFCAD2 数据集
    train_dataset = MFInstSegAdaptiveDataset(source_dir=source_dir, target_dir=target_dir, split='train',
                                             center_and_scale=True, normalize=False, num_threads=8, nums_data=100)

    print(f"Total dataset size: {len(train_dataset)}")
    source_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'source'])
    target_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'target'])
    print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")

    train_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)

    for data in train_loader:
        if data is not None:
            print("Batch data:")
            print(data)

            # 打印源域和目标域的语义分割标签
            print("Source Segmentation Labels:", data["source_graph"].ndata['seg_y'])
            print("Target Segmentation Labels:", data["target_graph"].ndata['seg_y'])
            break
# #MFCAD2->MFInst
# import pathlib
# import json
# import torch
# import dgl
# import numpy as np
# from torch.utils.data import DataLoader, Sampler
# from dataloader.base import BaseDataset
# from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
# import threading
#
#
# # 自定义批采样器，用于确保每个批次有相同数量的源域和目标域样本
# class DomainBatchSampler(Sampler):
#     def __init__(self, data_source, batch_size):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.source_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'source']
#         self.target_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'target']
#         self.num_batches = min(len(self.source_indices) // (batch_size // 2),
#                                len(self.target_indices) // (batch_size // 2))
#
#     def __iter__(self):
#         indices = []
#         for i in range(self.num_batches):
#             source_batch = self.source_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
#             target_batch = self.target_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
#             indices.extend(source_batch + target_batch)
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_batches * self.batch_size
#
#
# class MFInstSegAdaptiveDataset(BaseDataset):
#
#     @staticmethod
#     def num_classes(type='full'):
#         return 25 if type == 'full' else 7
#
#     def __init__(self,
#                  source_dir,  # 源域数据集
#                  target_dir,  # 目标域数据集
#                  graphs_source=None,
#                  graphs_target=None,
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
#         self.normalize = normalize
#         self.center_and_scale = center_and_scale
#         assert split in ("train", "val", "test", "all")
#
#         # 加载源域和目标域的数据集文件列表
#         split_filelist_source = self._load_split_file(self.source_path, split, nums_data)
#         split_filelist_target = self._load_split_file(self.target_path, split, nums_data)
#
#         print(f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split {split}")
#
#
#         print(f"Loading source graphs from {self.source_path}")
#         self.graphs_source = self.load_graphs(self.source_path, graphs_source, split_filelist_source, True, num_threads, dataset_type='MFCAD2')
#
#
#
#         print(f"Loading target graphs from {self.target_path}")
#         self.graphs_target = self.load_graphs(self.target_path, graphs_target, split_filelist_target, False, num_threads, dataset_type='MFInstSeg')
#
#
#         self.data_source = self.graphs_source
#         self.data_target = self.graphs_target
#         self.data = self.data_source + self.data_target
#         print(f"Done loading {len(self.data)} files for {split} split")
#
#     def _load_split_file(self, base_path, split, nums_data=None):
#         """
#         根据指定的 split 加载文件列表
#         """
#         split_txt_path = base_path / f'{split}.txt'
#         if not split_txt_path.exists():
#             raise FileNotFoundError(f"Could not find split file: {split_txt_path}")
#         with open(split_txt_path, 'r') as f:
#             split_filelist = [x.strip() for x in f.readlines()]
#
#         if nums_data is not None:
#             split_filelist = split_filelist[:nums_data]
#
#         return split_filelist
#
#     def load_graphs(self, file_path, graphs, split_file_list, has_labels, num_threads=4, dataset_type='MFCAD2'):
#         data = []
#         if graphs:
#             dataset = graphs
#         else:
#             dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
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
#             t = threading.Thread(target=lambda idx: results[idx].extend(
#                 self.process_chunk(
#                     chunks[idx], split_file_list,
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
#                 one_graph = self.load_one_graph(fn, data, dataset_type)
#                 if one_graph is None:
#                     continue
#                 if normalization_attribute and stat:
#                     one_graph = standardization(one_graph, stat)
#                 if center_and_scale_grid:
#                     one_graph = center_and_scale(one_graph)
#                 result.append(one_graph)
#         return result
#
#     def load_one_graph(self, fn, data, dataset_type):
#         """
#         加载单个图并附加语义分割标签，区分 MFCAD2 和 MFInstSeg 数据集的不同标签结构。
#         """
#         sample = load_one_graph(fn, data)
#
#         # 根据源域或目标域处理标签
#         label_file = self.source_path.joinpath("labels").joinpath(fn + ".json") if dataset_type == 'MFCAD2' else self.target_path.joinpath("labels").joinpath(fn + ".json")
#
#         try:
#             with open(str(label_file), "r") as read_file:
#                 labels_data = json.load(read_file)
#
#             # 处理 MFCAD2 数据集
#             if dataset_type == 'MFCAD2':
#                 if isinstance(labels_data, dict):
#                     labels_data = labels_data.get('seg', None)
#                 if labels_data is None or not isinstance(labels_data, (list, dict)):
#                     raise TypeError(f"Invalid label format in file {fn}")
#                 if isinstance(labels_data, dict):
#                     labels_data = [labels_data[str(face_id)] for face_id in range(sample['graph'].num_nodes())]
#
#             # 处理 MFInstSeg 数据集
#             elif dataset_type == 'MFInstSeg':
#                 if isinstance(labels_data, list):
#                     labels_data = labels_data[0][1]
#                 seg_labels = labels_data.get('seg', None)
#                 if seg_labels is None:
#                     raise ValueError(f"No 'seg' key found in {fn} for MFInstSeg dataset.")
#                 labels_data = [seg_labels[str(face_id)] for face_id in range(sample['graph'].num_nodes())]
#
#             # 将标签转换为张量
#             labels_data = np.array(labels_data, dtype=np.int32)
#             sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()
#
#         except (IndexError, TypeError, json.JSONDecodeError, ValueError) as e:
#             print(f"Error processing file {fn}: {e}")
#             return None
#
#         # 域区分
#         sample['domain'] = 'source' if dataset_type == 'MFCAD2' else 'target'
#         return sample
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#     def collate(self, batch):
#         source_graphs = []
#         target_graphs = []
#
#         # 将源域和目标域的图分别放入不同的列表
#         for sample in batch:
#             if sample['domain'] == 'source':
#                 source_graphs.append(sample['graph'])
#             else:
#                 target_graphs.append(sample['graph'])
#
#         # 分别对源域和目标域的图进行批处理
#         batched_source_graph = dgl.batch(source_graphs)
#         batched_target_graph = dgl.batch(target_graphs)
#
#         return {"source_graph": batched_source_graph,
#                 "target_graph": batched_target_graph}
#
#     def get_dataloader(self, batch_size=256, shuffle=False, sampler=None, num_workers=0, drop_last=True,
#                        pin_memory=False):
#         sampler = DomainBatchSampler(self.data, batch_size)
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
#     source_dir = 'E:\CHB\python_project\SFRGNN\MFCAD2'  # MFCAD2 数据集
#     target_dir = 'E:\CHB\python_project\SFRGNN\MFInstSeg'  # MFInstSeg 数据集
#     train_dataset = MFInstSegAdaptiveDataset(source_dir=source_dir, target_dir=target_dir, split='train',
#                                              center_and_scale=False, normalize=True, num_threads=8, nums_data=100)
#
#     print(f"Total dataset size: {len(train_dataset)}")
#     source_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'source'])
#     target_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'target'])
#     print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")
#
#     train_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)
#
#     for data in train_loader:
#         if data is not None:
#             print("Batch data:")
#             print(data)
#
#             # 打印源域和目标域的语义分割标签
#             print("Source Segmentation Labels:", data["source_graph"].ndata['seg_y'])
#             print("Target Segmentation Labels:", data["target_graph"].ndata['seg_y'])
#             break

# import pathlib
# import json
# import torch
# import dgl
# import numpy as np
# from torch.utils.data import DataLoader, Sampler
# from dataloader.base import BaseDataset
# from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
# import threading
#
#
# # 自定义批采样器，用于确保每个批次有相同数量的源域和目标域样本
# class DomainBatchSampler(Sampler):
#     def __init__(self, data_source, batch_size):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.source_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'source']
#         self.target_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'target']
#         self.num_batches = min(len(self.source_indices) // (batch_size // 2),
#                                len(self.target_indices) // (batch_size // 2))
#
#     def __iter__(self):
#         indices = []
#         for i in range(self.num_batches):
#             source_batch = self.source_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
#             target_batch = self.target_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
#             indices.extend(source_batch + target_batch)
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_batches * self.batch_size
#
#
# class MFInstSegAdaptiveDataset(BaseDataset):
#     @staticmethod
#     def num_classes(type='full'):
#         return 25 if type == 'full' else 7
#
#     def __init__(self,
#                  source_dir,  # 源域数据集
#                  target_dir,  # 目标域数据集
#                  graphs_source=None,
#                  graphs_target=None,
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
#         self.normalize = normalize
#         self.center_and_scale = center_and_scale
#         assert split in ("train", "val", "test", "all")
#
#         # 加载源域和目标域的数据集
#         split_filelist_source = self._load_split_file(self.source_path, split, nums_data)
#         split_filelist_target = self._load_split_file(self.target_path, split, nums_data)
#
#         print(
#             f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split {split}")
#
#         self.graphs_source = self.load_graphs(self.source_path, graphs_source, split_filelist_source, True, num_threads,
#                                               dataset_type='MFCAD2')
#         self.graphs_target = self.load_graphs(self.target_path, graphs_target, split_filelist_target, False,
#                                               num_threads, dataset_type='MFInstSeg')
#
#         self.data_source = self.graphs_source
#         self.data_target = self.graphs_target
#         self.data = self.data_source + self.data_target
#         print(f"Done loading {len(self.data)} files for {split} split")
#
#     def _load_split_file(self, base_path, split, nums_data=None):
#         """
#         根据指定的 split 加载文件列表
#         """
#         split_txt_path = base_path / f'{split}.txt'
#         if not split_txt_path.exists():
#             raise FileNotFoundError(f"Could not find split file: {split_txt_path}")
#         with open(split_txt_path, 'r') as f:
#             split_filelist = [x.strip() for x in f.readlines()]
#
#         if nums_data is not None:
#             split_filelist = split_filelist[:nums_data]
#
#         return split_filelist
#
#     def load_graphs(self, file_path, graphs, split_file_list, has_labels, num_threads=4, dataset_type='MFCAD2'):
#         self.data = []
#         if has_labels:
#             if graphs:
#                 self.dataset = graphs
#             else:
#                 self.dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
#         else:
#             if graphs:
#                 self.dataset = graphs
#             else:
#                 self.dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
#
#         if self.normalize:
#             stat = load_statistics(file_path.joinpath("aag").joinpath('attr_stat.json'))
#         else:
#             stat = None
#
#         chunk_size = (len(self.dataset) + num_threads - 1) // num_threads
#         chunks = [self.dataset[i:i + chunk_size] for i in range(0, len(self.dataset), chunk_size)]
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
#         self.data = [item for sublist in results for item in sublist]
#         return self.data
#
#     def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path,
#                       has_labels, dataset_type):
#         result = []
#         for one_data in chunk:
#             fn, data = one_data
#             if fn in split_file_list:
#                 one_graph = self.load_one_graph(fn, data, dataset_type)
#                 if one_graph is None:
#                     continue
#                 if normalization_attribute and stat:
#                     one_graph = standardization(one_graph, stat)
#                 if center_and_scale_grid:
#                     one_graph = center_and_scale(one_graph)
#                 result.append(one_graph)
#         return result
#
#     def load_one_graph(self, fn, data, dataset_type):
#         """
#         加载单个图并附加语义分割标签，区分 MFCAD2 和 MFInstSeg 数据集的不同标签结构。
#         """
#         sample = load_one_graph(fn, data)
#
#         # 根据源域或目标域处理标签
#         label_file = self.source_path.joinpath("labels").joinpath(
#             fn + ".json") if dataset_type == 'MFCAD2' else self.target_path.joinpath("labels").joinpath(fn + ".json")
#
#         try:
#             with open(str(label_file), "r") as read_file:
#                 labels_data = json.load(read_file)
#
#             # 处理 MFCAD2 数据集
#             if dataset_type == 'MFCAD2':
#                 # 检查是否为字典，且包含 'seg' 键
#                 if isinstance(labels_data, dict):
#                     labels_data = labels_data.get('seg', None)
#                 if labels_data is None or not isinstance(labels_data, (list, dict)):
#                     raise TypeError(f"Invalid label format in file {fn}")
#                 if isinstance(labels_data, dict):
#                     labels_data = [labels_data[str(face_id)] for face_id in range(sample['graph'].num_nodes())]
#
#             # 处理 MFInstSeg 数据集
#             elif dataset_type == 'MFInstSeg':
#                 if isinstance(labels_data, list):
#                     labels_data = labels_data[0][1]  # 处理嵌套结构
#                 seg_labels = labels_data.get('seg', None)
#                 if seg_labels is None:
#                     raise ValueError(f"No 'seg' key found in {fn} for MFInstSeg dataset.")
#                 labels_data = [seg_labels[str(face_id)] for face_id in range(sample['graph'].num_nodes())]
#
#             # 将标签转换为张量
#             labels_data = np.array(labels_data, dtype=np.int32)
#             sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()
#
#         except (IndexError, TypeError, json.JSONDecodeError, ValueError) as e:
#             print(f"Error processing file {fn}: {e}")
#             return None  # 跳过错误的文件
#
#         # 域区分
#         sample['domain'] = 'source' if dataset_type == 'MFCAD2' else 'target'
#         return sample
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#     def collate(self, batch):
#         source_graphs = []
#         target_graphs = []
#
#         # 将源域和目标域的图分别放入不同的列表
#         for sample in batch:
#             if sample['domain'] == 'source':
#                 source_graphs.append(sample['graph'])
#             else:
#                 target_graphs.append(sample['graph'])
#
#         # 分别对源域和目标域的图进行批处理
#         batched_source_graph = dgl.batch(source_graphs)
#         batched_target_graph = dgl.batch(target_graphs)
#
#         return {"source_graph": batched_source_graph,
#                 "target_graph": batched_target_graph}
#
#     def get_dataloader(self, batch_size=256, shuffle=False, sampler=None, num_workers=0, drop_last=True,
#                        pin_memory=False):
#         sampler = DomainBatchSampler(self.data, batch_size)
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
#     source_dir = '/mnt/data/CHB/AAGNet-main/MFCAD2'  # MFCAD2 数据集
#     target_dir = '/mnt/data/CHB/AAGNet-main/MFInstSeg'  # MFInstSeg 数据集
#     train_dataset = MFInstSegAdaptiveDataset(source_dir=source_dir, target_dir=target_dir, split='train',
#                                              center_and_scale=True, normalize=False, num_threads=8, nums_data=100)
#
#     print(f"Total dataset size: {len(train_dataset)}")
#     source_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'source'])
#     target_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'target'])
#     print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")
#
#     train_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)
#
#     for data in train_loader:
#         if data is not None:
#             print("Batch data:")
#             print(data)
#
#             # 打印源域和目标域的语义分割标签
#             print("Source Segmentation Labels:", data["source_graph"].ndata['seg_y'])
#             print("Target Segmentation Labels:", data["target_graph"].ndata['seg_y'])
#             break

#MF->MFCAD2
# import pathlib
# import json
# import torch
# import dgl
# import numpy as np
# import threading
# from torch.utils.data import DataLoader, Sampler
#
# from dataloader.base import BaseDataset
# from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
#
#
# # 自定义批采样器，用于确保每个批次有相同数量的源域和目标域样本
# class DomainBatchSampler(Sampler):
#     def __init__(self, data_source, batch_size):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.source_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'source']
#         self.target_indices = [i for i, sample in enumerate(data_source) if sample['domain'] == 'target']
#         self.num_batches = min(len(self.source_indices) // (batch_size // 2),
#                                len(self.target_indices) // (batch_size // 2))
#
#     def __iter__(self):
#         indices = []
#         for i in range(self.num_batches):
#             source_batch = self.source_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
#             target_batch = self.target_indices[i * (self.batch_size // 2):(i + 1) * (self.batch_size // 2)]
#             indices.extend(source_batch + target_batch)
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_batches * self.batch_size
#
#
# class MFInstSegAdaptiveDataset(BaseDataset):
#     # 使用缓存以数据集路径作为键
#     cached_graphs = {}
#
#     @staticmethod
#     def num_classes(type='full'):
#         return 25 if type == 'full' else 7
#
#     def __init__(self,
#                  source_dir,  # 源域数据集
#                  target_dir,  # 目标域数据集
#                  graphs_source=None,
#                  graphs_target=None,
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
#         self.normalize = normalize
#         self.center_and_scale = center_and_scale
#         assert split in ("train", "val", "test", "all")
#
#         # 加载源域和目标域的数据集文件列表
#         split_filelist_source = self._load_split_file(self.source_path, split, nums_data)
#         split_filelist_target = self._load_split_file(self.target_path, split, nums_data)
#
#         print(f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split {split}")
#
#         # 缓存源域和目标域的数据，只基于数据集路径进行缓存，不同split使用相同缓存
#         if str(self.source_path) in MFInstSegAdaptiveDataset.cached_graphs:
#             print(f"Using cached source graphs from {self.source_path}")
#             self.graphs_source = MFInstSegAdaptiveDataset.cached_graphs[str(self.source_path)]
#         else:
#             print(f"Loading source graphs from {self.source_path}")
#             self.graphs_source = self.load_graphs(self.source_path, graphs_source, split_filelist_source, True, num_threads, dataset_type='MFInstSeg')
#             MFInstSegAdaptiveDataset.cached_graphs[str(self.source_path)] = self.graphs_source
#
#         if str(self.target_path) in MFInstSegAdaptiveDataset.cached_graphs:
#             print(f"Using cached target graphs from {self.target_path}")
#             self.graphs_target = MFInstSegAdaptiveDataset.cached_graphs[str(self.target_path)]
#         else:
#             print(f"Loading target graphs from {self.target_path}")
#             self.graphs_target = self.load_graphs(self.target_path, graphs_target, split_filelist_target, False, num_threads, dataset_type='MFCAD2')
#             MFInstSegAdaptiveDataset.cached_graphs[str(self.target_path)] = self.graphs_target
#
#         self.data_source = self.graphs_source
#         self.data_target = self.graphs_target
#         self.data = self.data_source + self.data_target
#         print(f"Done loading {len(self.data)} files for {split} split")
#
#     def _load_split_file(self, base_path, split, nums_data=None):
#         """
#         根据指定的 split 加载文件列表
#         """
#         split_txt_path = base_path / f'{split}.txt'
#         if not split_txt_path.exists():
#             raise FileNotFoundError(f"Could not find split file: {split_txt_path}")
#         with open(split_txt_path, 'r') as f:
#             split_filelist = [x.strip() for x in f.readlines()]
#
#         if nums_data is not None:
#             split_filelist = split_filelist[:nums_data]
#
#         return split_filelist
#
#     def load_graphs(self, file_path, graphs, split_file_list, has_labels, num_threads=4, dataset_type='MFInstSeg'):
#         data = []
#         if graphs:
#             dataset = graphs
#         else:
#             dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
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
#             t = threading.Thread(target=lambda idx: results[idx].extend(
#                 self.process_chunk(
#                     chunks[idx], split_file_list,
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
#                 one_graph = self.load_one_graph(fn, data, dataset_type)
#                 if one_graph is None:
#                     continue
#                 if normalization_attribute and stat:
#                     one_graph = standardization(one_graph, stat)
#                 if center_and_scale_grid:
#                     one_graph = center_and_scale(one_graph)
#                 result.append(one_graph)
#         return result
#
#     def load_one_graph(self, fn, data, dataset_type):
#         """
#         加载单个图并附加语义分割标签，区分 MFInstSeg 和 MFCAD2 数据集的不同标签结构。
#         """
#         sample = load_one_graph(fn, data)
#
#         # 根据源域或目标域处理标签
#         if dataset_type == 'MFInstSeg':
#             label_file = self.source_path.joinpath("labels").joinpath(fn + ".json")
#         else:
#             label_file = self.target_path.joinpath("labels").joinpath(fn + ".json")
#
#         try:
#             with open(str(label_file), "r") as read_file:
#                 labels_data = json.load(read_file)
#
#             if dataset_type == 'MFInstSeg':
#                 # 处理 MFInstSeg 数据集的标签
#                 if isinstance(labels_data, list):
#                     labels_data = labels_data[0][1]  # 处理嵌套结构
#                 seg_labels = labels_data.get('seg', None)
#                 if seg_labels is None:
#                     raise ValueError(f"No 'seg' key found in {fn} for MFInstSeg dataset.")
#                 labels_data = [seg_labels[str(face_id)] for face_id in range(sample['graph'].num_nodes())]
#             else:
#                 # 处理 MFCAD2 数据集的标签
#                 if isinstance(labels_data, dict):
#                     labels_data = labels_data.get('seg', None)
#                 if labels_data is None or not isinstance(labels_data, (list, dict)):
#                     raise TypeError(f"Invalid label format in file {fn}")
#                 if isinstance(labels_data, dict):
#                     labels_data = [labels_data[str(face_id)] for face_id in range(sample['graph'].num_nodes())]
#
#             # 将标签转换为张量
#             labels_data = np.array(labels_data, dtype=np.int32)
#             sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()
#
#         except (IndexError, TypeError, json.JSONDecodeError, ValueError) as e:
#             print(f"Error processing file {fn}: {e}")
#             return None  # 跳过错误的文件
#
#         # 域区分
#         sample['domain'] = 'source' if dataset_type == 'MFInstSeg' else 'target'
#         return sample
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#     def collate(self, batch):
#         source_graphs = []
#         target_graphs = []
#
#         # 将源域和目标域的图分别放入不同的列表
#         for sample in batch:
#             if sample['domain'] == 'source':
#                 source_graphs.append(sample['graph'])
#             else:
#                 target_graphs.append(sample['graph'])
#
#         # 分别对源域和目标域的图进行批处理
#         batched_source_graph = dgl.batch(source_graphs)
#         batched_target_graph = dgl.batch(target_graphs)
#
#         return {"source_graph": batched_source_graph,
#                 "target_graph": batched_target_graph}
#
#     def get_dataloader(self, batch_size=256, shuffle=False, sampler=None, num_workers=0, drop_last=True,
#                        pin_memory=False):
#         sampler = DomainBatchSampler(self.data, batch_size)
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
#     source_dir = '/mnt/data/CHB/AAGNet-main/MFInstSeg'  # 源域数据集
#     target_dir = '/mnt/data/CHB/AAGNet-main/MFCAD2'     # 目标域数据集
#
#     # 初始化训练集
#     train_dataset = MFInstSegAdaptiveDataset(
#         source_dir=source_dir,
#         target_dir=target_dir,
#         split='train',
#         center_and_scale=True,
#         normalize=False,
#         num_threads=8,
#         nums_data=100
#     )
#
#     print(f"Total training dataset size: {len(train_dataset)}")
#     source_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'source'])
#     target_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'target'])
#     print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")
#
#     train_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)
#
#     # 初始化验证集，复用缓存的图数据
#     val_dataset = MFInstSegAdaptiveDataset(
#         source_dir=source_dir,
#         target_dir=target_dir,
#         split='val',
#         center_and_scale=True,
#         normalize=False,
#         num_threads=8,
#         nums_data=100
#     )
#
#     print(f"Total validation dataset size: {len(val_dataset)}")
#     val_loader = val_dataset.get_dataloader(batch_size=8, pin_memory=True)
#
#     # 测试训练数据加载
#     for data in train_loader:
#         if data is not None:
#             print("Training Batch data:")
#             print(data)
#             print("Source Segmentation Labels:", data["source_graph"].ndata['seg_y'])
#             print("Target Segmentation Labels:", data["target_graph"].ndata['seg_y'])
#             break
#
#     # 测试验证数据加载
#     for data in val_loader:
#         if data is not None:
#             print("Validation Batch data:")
#             print(data)
#             print("Source Segmentation Labels:", data["source_graph"].ndata['seg_y'])
#             print("Target Segmentation Labels:", data["target_graph"].ndata['seg_y'])
#             break
#
