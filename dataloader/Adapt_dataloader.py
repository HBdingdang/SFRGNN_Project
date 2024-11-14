
import pathlib
import json
import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader, Sampler
from dataloader.base import BaseDataset
from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
import threading

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
                 source_dir,
                 target_dir,
                 graphs_source=None,
                 graphs_target=None,
                 split="train",
                 normalize=True,
                 center_and_scale=True,
                 random_rotate=False,
                 nums_data=None,
                 transform=None,
                 dataset_type='MFCAD2',  # 'MFCAD' 或 'MFCAD2'
                 num_threads=0):
        super().__init__(transform, random_rotate)
        self.source_path = pathlib.Path(source_dir)
        self.target_path = pathlib.Path(target_dir)
        self.dataset_type = dataset_type
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        assert split in ("train", "val", "test", "all")

        # 加载源域和目标域的数据集
        split_filelist_source = self._load_split_file(self.source_path, split, nums_data)
        split_filelist_target = self._load_split_file(self.target_path, split, nums_data)

        print(f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split {split}")

        self.graphs_source = self.load_graphs(self.source_path, graphs_source, graphs_target, split_filelist_source, True, num_threads, dataset_type='MFCAD')
        self.graphs_target = self.load_graphs(self.target_path, graphs_source, graphs_target, split_filelist_target, False, num_threads, dataset_type='MFCAD2')

        self.data_source = self.graphs_source
        self.data_target = self.graphs_target
        self.data = self.data_source + self.data_target
        print(f"Done loading {len(self.data)} files for {split} split")

    def _load_split_file(self, base_path, split, nums_data=None):
        """
        根据文件结构分别加载 MFCAD 和 MFCAD2 的数据集分割文件。
        MFCAD 使用 split.json，MFCAD2 使用 .txt 文件。
        """
        # 检查是否有 split.json 文件（MFCAD）
        split_json_path = base_path / 'split.json'
        if split_json_path.exists():
            with open(split_json_path, 'r') as f:
                split_data = json.load(f)
            if split == 'train':
                split_filelist = split_data['train']
            elif split == 'val':
                split_filelist = split_data['validation']
            else:
                split_filelist = split_data['test']
        else:
            # MFCAD2 数据集使用 .txt 文件
            split_txt_path = base_path / f'{split}.txt'
            if not split_txt_path.exists():
                raise FileNotFoundError(f"Could not find split file: {split_txt_path}")
            with open(split_txt_path, 'r') as f:
                split_filelist = [x.strip() for x in f.readlines()]

        if nums_data is not None:
            split_filelist = split_filelist[:nums_data]

        return split_filelist

    def load_graphs(self, file_path, source_graphs, target_graphs, split_file_list, has_labels, num_threads=4, dataset_type='MFCAD'):
        self.data = []
        if has_labels:
            if source_graphs:
                self.source_dataset = source_graphs
            else:
                self.source_dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
        else:
            if target_graphs:
                self.target_dataset = target_graphs
            else:
                self.target_dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))

        if self.normalize:
            stat = load_statistics(file_path.joinpath("aag").joinpath('attr_stat.json'))
        else:
            stat = None

        self.dataset = self.source_dataset if has_labels else self.target_dataset

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
        加载单个文件的数据。

        Args:
            fn (str): 文件名。
            data (dict): 文件数据。
            dataset_type (str): 数据集类型，'MFCAD' 或 'MFCAD2'。

        Returns:
            dict: 包含图数据和标签的样本。
        """
        # 加载图的基础数据
        sample = load_one_graph(fn, data)

        # 根据不同数据集加载标签
        if dataset_type == 'MFCAD':
            # MFCAD 的标签文件路径和加载逻辑
            label_file = self.source_path.joinpath("labels").joinpath(fn + "_color_ids.json")
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)

            label = []
            for face in labels_data["body"]["faces"]:
                index = face["segment"]["index"]
                label.append(index)

            # 将标签存储到图的节点数据中
            sample["graph"].ndata["seg_y"] = torch.tensor(label).long()

        elif dataset_type == 'MFCAD2':
            # MFCAD2 的标签文件路径和加载逻辑
            label_file = self.target_path.joinpath("labels").joinpath(fn + ".json")
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)

            # MFCAD2 直接将整个标签数组加载为 NumPy 数组并转换为 tensor
            labels_data = np.array(labels_data, dtype=np.int32)

            # 将标签存储到图的节点数据中
            sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()

        # 标记域
        sample['domain'] = 'source' if dataset_type == 'MFCAD' else 'target'
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        source_graphs = []
        target_graphs = []

        # 将源域和目标域的图分别放入对应的列表
        for sample in batch:
            if sample['domain'] == 'source':
                source_graphs.append(sample['graph'])
            else:
                target_graphs.append(sample['graph'])

        # 批处理源域和目标域的图
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
    source_dir = '/mnt/data/CHB/AAGNet-main/MFCAD'  # MFCAD 数据集
    target_dir = '/mnt/data/CHB/AAGNet-main/MFCAD2' # MFCAD2 数据集
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
