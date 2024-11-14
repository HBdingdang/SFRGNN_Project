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
        self.num_batches = min(self.data_source_size // self.half_batch_size,
                               self.data_target_size // self.half_batch_size)

    def __iter__(self):
        indices = []
        for i in range(self.num_batches):
            source_indices = list(range(i * self.half_batch_size, (i + 1) * self.half_batch_size))
            target_indices = list(range(i * self.half_batch_size, (i + 1) * self.half_batch_size))
            batch_indices = source_indices + target_indices
            indices.extend(batch_indices)

        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size


class MFInstSegAdaptiveDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 25

    def __init__(self,
                 source_dir,
                 target_dir,
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

        # 加载文件列表
        self.source_file_paths = self._load_files(self.source_path, split, nums_data)
        self.target_file_paths = self._load_files(self.target_path, split, nums_data)
        print(f"Loaded {len(self.source_file_paths)} source files and {len(self.target_file_paths)} target files for split {split}")

        # 加载源域和目标域的图数据
        self.graphs_source = self.load_graphs(self.source_path, self.source_file_paths, True, num_threads)
        self.graphs_target = self.load_graphs(self.target_path, self.target_file_paths, False, num_threads)

        self.data_source = self.graphs_source
        self.data_target = self.graphs_target

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

        if nums_data is not None:
            split_filelist = split_filelist[:nums_data]

        return split_filelist

    def load_graphs(self, file_path, split_file_list, has_labels, num_threads=4):
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
                    self.normalize, self.center_and_scale, stat, file_path, has_labels)), args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        data = sorted([item for sublist in results for item in sublist], key=lambda x: x['filename'])
        return data

    def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path,
                      has_labels):
        result = []
        for one_data in chunk:
            fn, data = one_data
            if fn in split_file_list:
                one_graph = self.load_one_graph(fn, data, has_labels)

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

    def load_one_graph(self, fn, data, has_labels):
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

        face_segmentation_labels = np.array([seg_label[str(face_id)] for face_id in range(num_faces)])
        instance_label = np.array(inst_label, dtype=np.int32)
        bottom_segmentation_labels = np.array([bottom_label[str(face_id)] for face_id in range(num_faces)])

        sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentation_labels).long()
        sample["inst_y"] = torch.tensor(instance_label).float()
        sample["graph"].ndata["bottom_y"] = torch.tensor(bottom_segmentation_labels).float().reshape(-1, 1)
        sample['domain'] = 'source' if has_labels else 'target'
        sample["graph"].ndata['domain'] = torch.tensor([1 if has_labels else 0] * num_faces, dtype=torch.long)

        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of range for data_source or data_target with length {len(self.data_source)} and {len(self.data_target)}.")

        source_idx = idx % len(self.data_source)
        target_idx = idx % len(self.data_target)

        source_data = self.data_source[source_idx]
        target_data = self.data_target[target_idx]

        return {"source_data": source_data, "target_data": target_data}

    def __len__(self):
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


if __name__ == "__main__":
    source_dir = 'E:\CHB\python_project\SFRGNN\MFInstSeg'
    target_dir = 'E:\CHB\python_project\SFRGNN\dataset\data_3w'

    # 初始化数据集
    dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        split='test',  # 测试集
        normalize=True,
        center_and_scale=False,
        num_threads=8,
        nums_data=100  # 仅加载前100个样本以便快速测试
    )

    # 创建DataLoader
    dataloader = dataset.get_dataloader(batch_size=10, pin_memory=True)

    print("DataLoader initialized successfully.")
    print(f"Total dataset length: {len(dataset)}")

    # 遍历 DataLoader 进行测试，只打印第一个批次的数据
    for batch_idx, data in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")

        source_graphs = data['source_graph']
        target_graphs = data['target_graph']
        source_inst_labels = data['source_inst_labels']
        target_inst_labels = data['target_inst_labels']

        # 打印图信息
        print(f"Source Graphs: {source_graphs}")
        print(f"Target Graphs: {target_graphs}")

        # 打印实例分割标签形状
        print(f"Source Instance Labels Shape: {source_inst_labels.shape}")
        print(f"Target Instance Labels Shape: {target_inst_labels.shape}")

        # 打印语义分割标签
        print("Source Graphs Semantic Segmentation Labels:")
        print(source_graphs.ndata['seg_y'])

        print("Target Graphs Semantic Segmentation Labels:")
        print(target_graphs.ndata['seg_y'])

        # 打印实例分割标签
        print("Source Graphs Instance Segmentation Labels:")
        print(source_inst_labels)

        print("Target Graphs Instance Segmentation Labels:")
        print(target_inst_labels)

        # 只打印一个批次的数据
        break
