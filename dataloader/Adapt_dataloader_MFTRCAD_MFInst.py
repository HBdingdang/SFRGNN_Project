import os
import json
import pathlib
import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader, Sampler
from dataloader.base import BaseDataset
from utils.data_utils import load_one_graph as base_load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
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
                 source_dir,  # MFTRCAD 数据集
                 target_dir,  # MFInstSeg 数据集
                 source_dataset_type,  # 源域数据集类型，例如 'MFTRCAD'
                 target_dataset_type,  # 目标域数据集类型，例如 'MFInstSeg'
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
        self.source_dataset_type = source_dataset_type
        self.target_dataset_type = target_dataset_type
        assert split in ("train", "val", "test", "all"), "Split must be one of 'train', 'val', 'test', 'all'"

        # 加载源域 (MFTRCAD) 和目标域 (MFInstSeg) 的数据集
        split_filelist_source = self._load_split_file(self.source_path, split, nums_data)
        split_filelist_target = self._load_split_file(self.target_path, split, nums_data)

        print(f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split '{split}'")

        # 加载源域数据
        self.graphs_source = self.load_graphs(self.source_path, graphs_source, split_filelist_source, has_labels=True, num_threads=num_threads, dataset_type=source_dataset_type)

        # 加载目标域数据
        self.graphs_target = self.load_graphs(self.target_path, graphs_target, split_filelist_target, has_labels=True, num_threads=num_threads, dataset_type=target_dataset_type)

        self.data_source = self.graphs_source
        self.data_target = self.graphs_target
        self.data = self.data_source + self.data_target
        print(f"Done loading {len(self.data)} files for split '{split}'")

    def _load_split_file(self, base_path, split, nums_data=None):
        """
        根据指定的 split 加载 MFTRCAD 或 MFInstSeg 数据集的文件列表
        """
        split_txt_path = base_path / f'{split}.txt'
        if not split_txt_path.exists():
            raise FileNotFoundError(f"Could not find split file: {split_txt_path}")
        with open(split_txt_path, 'r') as f:
            split_filelist = [x.strip() for x in f.readlines()]

        if nums_data is not None:
            split_filelist = split_filelist[:nums_data]

        return split_filelist

    def load_graphs(self, file_path, graphs, split_file_list, has_labels, num_threads=4, dataset_type=None):
        """
        加载图数据，并处理标签文件的存在性
        """
        if graphs:
            self.dataset = graphs
        else:
            graphs_json_path = file_path.joinpath("aag").joinpath('graphs.json')
            self.dataset = load_json_or_pkl(graphs_json_path)

        if self.normalize:
            stat_path = file_path.joinpath("aag").joinpath('attr_stat.json')
            stat = load_statistics(stat_path)
        else:
            stat = None

        # 分割数据以支持多线程
        if isinstance(self.dataset, dict):
            data_iter = list(self.dataset.items())  # List of (fn, data)
        elif isinstance(self.dataset, list):
            data_iter = self.dataset  # List of [fn, data] 或其他格式
        else:
            raise ValueError("Unsupported dataset structure. Expected dict or list.")

        chunk_size = (len(data_iter) + num_threads - 1) // num_threads
        chunks = [data_iter[i:i + chunk_size] for i in range(0, len(data_iter), chunk_size)]

        threads = []
        results = [[] for _ in range(num_threads)]
        for i in range(num_threads):
            t = threading.Thread(target=lambda i: results[i].extend(
                self.process_chunk(
                    chunks[i], split_file_list=split_file_list,
                    normalization_attribute=self.normalize,
                    center_and_scale_grid=self.center_and_scale,
                    stat=stat, file_path=file_path, dataset_type=dataset_type
                )), args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 合并所有线程的结果
        self.data = [item for sublist in results for item in sublist if item is not None]
        return self.data


    def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path, dataset_type):
        result = []
        for one_data in chunk:
            if isinstance(one_data, tuple):
                fn, data = one_data
            elif isinstance(one_data, list) and len(one_data) >= 2:
                fn, data = one_data[0], one_data[1]
            else:
                print(f"Warning: Unexpected data format {one_data}. Skipping.")
                continue

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
        加载单个图并附加语义分割标签，基于 dataset_type 加载标签文件。
        """
        sample = base_load_one_graph(fn, data)
        domain = None
        # 根据 dataset_type 确定加载标签文件的路径
        if dataset_type == self.source_dataset_type:
            label_file = self.source_path.joinpath("labels").joinpath(fn + ".json")
            current_dataset_type = self.source_dataset_type
            domain = 'source'
        elif dataset_type == self.target_dataset_type:
            label_file = self.target_path.joinpath("labels").joinpath(fn + ".json")
            current_dataset_type = self.target_dataset_type
            domain = 'target'
        else:
            print(f"Warning: Unknown dataset_type '{dataset_type}' for file {fn}. Skipping.")
            return None

        if not label_file.exists():
            print(f"Warning: Label file {label_file} does not exist. Skipping.")
            return None

        try:
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for {label_file}: {e}. Skipping.")
            return None

        if current_dataset_type == 'MFTRCAD':
            # 处理源域数据集标签格式
            if isinstance(labels_data, list):
                # 假设每个索引代表一个面的标签
                seg_label = {str(i): label for i, label in enumerate(labels_data)}
            elif isinstance(labels_data, dict):
                seg_label = labels_data.get("cls", {})
            else:
                print(f"Warning: Unsupported label format in {label_file}. Skipping.")
                return None

            if not isinstance(seg_label, dict):
                print(f"Warning: 'cls' in {label_file} is not a dict. Skipping.")
                return None

            num_faces = sample['graph'].num_nodes()
            if num_faces != len(seg_label):
                print(f"Warning: File {fn} has {len(seg_label)} labels but graph has {num_faces} faces. Skipping.")
                return None

            # 将超过24的类别标签转换为24
            face_segmentation_labels = np.zeros(num_faces)
            for face_id in range(num_faces):
                label = seg_label.get(str(face_id), 0)  # 默认标签为0
                face_segmentation_labels[face_id] = min(label, 24)  # 超过24的标签转换为24
            # 添加标签到图的节点数据
            sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentation_labels).long()

        elif current_dataset_type == 'MFInstSeg':
            # 处理 MFInstSeg 数据集标签格式
            seg_labels = {}
            if isinstance(labels_data, list):
                for entry in labels_data:
                    if isinstance(entry, dict):
                        seg = entry.get('seg', {})
                        if isinstance(seg, dict):
                            seg_labels.update(seg)
                    elif isinstance(entry, list) or isinstance(entry, tuple):
                        if len(entry) > 1 and isinstance(entry[1], dict):
                            seg = entry[1].get('seg', {})
                            if isinstance(seg, dict):
                                seg_labels.update(seg)
            elif isinstance(labels_data, dict):
                seg_labels = labels_data.get('seg', {})
                if not isinstance(seg_labels, dict):
                    print(f"Warning: 'seg' in {label_file} is not a dict. Skipping.")
                    return None
            else:
                print(f"Warning: Unexpected label format in {label_file}. Skipping.")
                return None

            num_faces = sample['graph'].num_nodes()
            if num_faces != len(seg_labels):
                print(f"Warning: File {fn} has {len(seg_labels)} labels but graph has {num_faces} faces. Skipping.")
                return None
            # 将标签转换为张量
            face_segmentation_labels = np.zeros(num_faces)
            for face_id in range(num_faces):
                label = seg_labels.get(str(face_id), 0)  # 默认标签为0
                face_segmentation_labels[face_id] = label
            # 添加标签到图的节点数据
            sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentation_labels).long()

        else:
            print(f"Warning: Unknown dataset_type '{current_dataset_type}' for file {fn}. Skipping.")
            return None

        sample['domain'] = domain
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        """
        Collate a batch of data samples together into a single batch.
        Ensures that each batch has equal source and target samples.
        """
        source_graphs = []
        target_graphs = []
        batched_filenames = []

        for sample in batch:
            if sample['domain'] == 'source':
                source_graphs.append(sample['graph'])
            elif sample['domain'] == 'target':
                target_graphs.append(sample['graph'])
            batched_filenames.append(sample['filename'])

        # 批处理源域和目标域的图
        batched_source_graph = dgl.batch(source_graphs) if source_graphs else None
        batched_target_graph = dgl.batch(target_graphs) if target_graphs else None

        return {
            "source_graph": batched_source_graph,
            "target_graph": batched_target_graph,
            "filename": batched_filenames
        }

    def get_dataloader(self, batch_size=256, shuffle=False, sampler=None, num_workers=0, drop_last=True, pin_memory=False):
        if sampler is None:
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
    source_dir = 'E:\CHB\python_project\SFRGNN\MFTRCAD'  # MFTRCAD 数据集
    target_dir = 'E:\CHB\python_project\SFRGNN\MFInstSeg'  # MFInstSeg 数据集
    print("start 01")
    train_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        source_dataset_type='MFTRCAD',
        target_dataset_type='MFInstSeg',
        split='train',  # 可以根据需要设置 'train', 'val', 'test', 'all'
        center_and_scale=False,
        normalize=True,
        random_rotate=False,
        num_threads=8,
        nums_data=1000  # 可以设置具体的数字限制加载的数据量
    )
    print("start 02")
    val_dataset = MFInstSegAdaptiveDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        source_dataset_type='MFTRCAD',
        target_dataset_type='MFInstSeg',
        split='test',  # 可以根据需要设置 'train', 'val', 'test', 'all'
        center_and_scale=False,
        normalize=True,
        random_rotate=False,
        num_threads=8,
        nums_data=800  # 可以设置具体的数字限制加载的数据量
    )

    print(f"Total dataset size: {len(val_dataset)}")
    source_data_size = len([sample for sample in val_dataset.data if sample['domain'] == 'source'])
    target_data_size = len([sample for sample in val_dataset.data if sample['domain'] == 'target'])
    print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")

    try:
        val_loader = val_dataset.get_dataloader(batch_size=8, pin_memory=True)

        for data in val_loader:
            if data is not None:
                print("Batch data:")
                print(data)

                # 打印源域和目标域的语义分割标签
                if data["source_graph"] is not None:
                    print("Source Segmentation Labels:", data["source_graph"].ndata['seg_y'])
                if data["target_graph"] is not None:
                    print("Target Segmentation Labels:", data["target_graph"].ndata['seg_y'])
                break  # 只打印第一个批次
    except ValueError as e:
        print(f"DataLoader error: {e}")
