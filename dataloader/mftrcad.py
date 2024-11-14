import os
import json
import pathlib
import torch
import dgl
import numpy as np
import threading

from torch.utils.data import DataLoader
from dataloader.base import BaseDataset
from utils.data_utils import (
    load_one_graph as base_load_one_graph,
    load_json_or_pkl,
    load_statistics,
    standardization,
    center_and_scale
)


class MFTRCADSegDataset(BaseDataset):
    def __init__(self,
                 root_dir,
                 graphs=None,
                 split="train",
                 normalize=True,
                 center_and_scale=True,
                 random_rotate=False,
                 nums_data=-1,
                 transform=None,
                 num_threads=0):
        """
        从根目录加载 MFTRCADSegDataset 数据集。
        Args:
            root_dir (str): 数据集的根路径。
            graphs (list, optional): 预加载的图数据列表。
            split (str, optional): 要加载的数据划分。默认为 "train"。
            normalize (bool, optional): 是否对数据进行标准化。默认为 True。
            center_and_scale (bool, optional): 是否对实体进行中心化和缩放。默认为 True。
            random_rotate (bool, optional): 是否对实体进行 90 度增量的随机旋转。默认为 False。
            nums_data (int, optional): 要使用的数据示例数量。默认为 -1（使用所有数据）。
            transform (callable, optional): 要应用于数据的转换。
            num_threads (int, optional): 用于数据加载的线程数量。默认为 0。
        """
        super().__init__(transform, random_rotate)
        path = pathlib.Path(root_dir)
        self.path = path
        self.transform = transform
        self.random_rotate = random_rotate
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        self.num_threads = num_threads
        self.split = split

        assert split in ("train", "val", "test", "all"), "Split must be one of 'train', 'val', 'test', 'all'"

        # 从对应的 txt 文件（train.txt, val.txt, test.txt）加载数据分区
        filelist_path = path / f'{split}.txt'
        if not filelist_path.exists():
            raise FileNotFoundError(f"Split file {filelist_path} does not exist.")

        with open(filelist_path, 'r') as f:
            split_filelist = f.readlines()

        if nums_data != -1:
            split_filelist = split_filelist[:nums_data]

        split_filelist = [x.strip() for x in split_filelist]  # 移除任何多余的空白字符

        # 尝试加载每个图并过滤掉缺失的标签文件
        valid_filelist = []
        for filename in split_filelist:
            label_file = path / "labels" / f"{filename}.json"
            if not label_file.exists():
                print(f"Warning: Label file {label_file} does not exist. Skipping.")
                continue
            valid_filelist.append(filename)

        # 如果有文件被跳过，更新 split 文件（例如 test.txt）
        if len(valid_filelist) < len(split_filelist):
            with open(filelist_path, 'w') as f:
                for item in valid_filelist:
                    f.write(f"{item}\n")
            print(f"Updated {split}.txt with valid files. Removed {len(split_filelist) - len(valid_filelist)} entries.")

        # 加载有效文件的图
        print(f"Loading {split} data...")
        split_filelist = set(valid_filelist)
        graph_path = path / "aag"
        # 加载图
        self.data = self.load_graphs(graph_path, graphs, split_filelist)

        print(f"Done loading {len(self.data)} files for split '{split}'.")

    def load_graphs(self, graph_path, graphs, split_filelist):
        """
        加载图数据，并处理标签文件的存在性和标签最大值限制。
        Args:
            graph_path (Path): 包含 'graphs.json' 或 'graphs.pkl' 的 'aag' 目录路径。
            graphs (list, optional): 预加载的图数据。
            split_filelist (set): 要包含的文件名集合。
        Returns:
            list: 加载和处理后的图样本列表。
        """
        if graphs:
            self.dataset = graphs
        else:
            # 尝试首先加载 'graphs.pkl'，否则加载 'graphs.json'
            graphs_pkl_path = graph_path / 'graphs.pkl'
            graphs_json_path = graph_path / 'graphs.json'
            if graphs_pkl_path.exists():
                print(f"Loading data from {graphs_pkl_path}")
                self.dataset = load_json_or_pkl(graphs_pkl_path)
            elif graphs_json_path.exists():
                print(f"Loading data from {graphs_json_path}")
                self.dataset = load_json_or_pkl(graphs_json_path)
            else:
                raise FileNotFoundError(f"Neither {graphs_pkl_path} nor {graphs_json_path} exists.")

        # 如果需要标准化，加载统计信息
        if self.normalize:
            stat_path = graph_path / 'attr_stat.json'
            if not stat_path.exists():
                raise FileNotFoundError(f"Statistics file {stat_path} does not exist.")
            stat = load_statistics(stat_path)
        else:
            stat = None

        # 准备数据迭代器
        if isinstance(self.dataset, dict):
            data_iter = list(self.dataset.items())  # List of (fn, data)
        elif isinstance(self.dataset, list):
            data_iter = self.dataset  # List of [fn, data] 或其他格式
        else:
            raise ValueError("Unsupported dataset structure. Expected dict or list.")

        # 将数据分割成多个块，以支持多线程
        chunk_size = (len(data_iter) + self.num_threads - 1) // self.num_threads if self.num_threads > 0 else len(data_iter)
        if self.num_threads > 0:
            chunks = [data_iter[i:i + chunk_size] for i in range(0, len(data_iter), chunk_size)]
        else:
            chunks = [data_iter]  # Single chunk if no threading

        # 多线程处理
        if self.num_threads > 0:
            threads = []
            results = [[] for _ in range(self.num_threads)]
            for i in range(self.num_threads):
                t = threading.Thread(target=lambda i: results[i].extend(
                    self.process_chunk(
                        chunks[i], split_filelist,
                        self.normalize, self.center_and_scale, stat
                    )), args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # 合并所有线程的结果
            processed_data = [item for sublist in results for item in sublist if item is not None]
        else:
            # Single-threaded processing
            processed_data = self.process_chunk(data_iter, split_filelist, self.normalize, self.center_and_scale, stat)

        return processed_data

    def process_chunk(self, chunk, split_filelist, normalization_attribute, center_and_scale_grid, stat):
        """
        处理数据块中的每个样本。
        Args:
            chunk (list): 数据块。
            split_filelist (set): 要包含的文件名集合。
            normalization_attribute (bool): 是否进行标准化。
            center_and_scale_grid (bool): 是否进行中心化和缩放。
            stat (dict, optional): 统计信息，用于标准化。
        Returns:
            list: 处理后的样本列表。
        """
        result = []
        for one_data in chunk:
            if isinstance(one_data, tuple) or isinstance(one_data, list):
                fn, data = one_data[0], one_data[1]
            else:
                print(f"Warning: Unexpected data format {one_data}. Skipping.")
                continue

            if fn in split_filelist:
                one_graph = self.load_one_graph(fn, data)
                if one_graph is None:
                    continue
                if normalization_attribute and stat:
                    one_graph = standardization(one_graph, stat)
                if center_and_scale_grid:
                    one_graph = center_and_scale(one_graph)
                result.append(one_graph)
        return result

    def load_one_graph(self, fn, data):
        """
        加载单个文件的数据并附加语义分割标签。
        Args:
            fn (str): 文件名。
            data (dict): 文件的数据。
        Returns:
            dict: 文件的数据或如果标签缺失则为 None。
        """
        # 使用基类方法加载图
        sample = base_load_one_graph(fn, data)
        num_faces = sample['graph'].num_nodes()

        # 加载并检查标签文件
        label_file = self.path / "labels" / f"{fn}.json"
        if not label_file.exists():
            print(f"Warning: Label file {label_file} does not exist. Skipping.")
            return None

        with open(str(label_file), "r") as read_file:
            try:
                labels_data = json.load(read_file)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for {label_file}: {e}. Skipping.")
                return None

        seg_label = labels_data.get("cls", {})
        if not isinstance(seg_label, dict):
            print(f"Warning: 'cls' in {label_file} is not a dict. Skipping.")
            return None

        if num_faces != len(seg_label):
            print(f"File {fn} has {len(seg_label)} labels but graph has {num_faces} faces. Skipping.")
            return None

        # 将标签转换为张量，并确保所有标签不超过 24
        face_segmentation_labels = np.zeros(num_faces)
        for face_id in range(num_faces):
            label = seg_label.get(str(face_id), 0)  # 如果缺失则默认标签为 0
            face_segmentation_labels[face_id] = min(label, 24)  # 将超过 24 的标签转换为 24

        # 将转换后的标签附加到图的节点数据中
        sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentation_labels).long()
        sample["filename"] = fn  # 确保 'filename' 键存在于样本中

        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        """
        将一批数据样本聚合成一个批次。
        Args:
            batch (List[dict]): 数据样本列表。
        Returns:
            dict: 批处理后的数据。
        """
        # 聚合图数据
        batched_graph = dgl.batch([sample["graph"] for sample in batch]) if batch else None

        # 收集每个图的文件名
        batched_filenames = [sample["filename"] for sample in batch] if batch else []

        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self, batch_size=256, shuffle=False, num_workers=0, drop_last=True, pin_memory=False):
        """
        获取数据加载器。
        Args:
            batch_size (int, optional): 每批次的样本数量。默认为 256。
            shuffle (bool, optional): 是否打乱数据。默认为 False。
            num_workers (int, optional): 使用的工作线程数量。默认为 0。
            drop_last (bool, optional): 是否丢弃最后一个不完整的批次。默认为 True。
            pin_memory (bool, optional): 是否固定内存。默认为 False。
        Returns:
            DataLoader: 配置好的数据加载器。
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )


# 使用示例
if __name__ == '__main__':
    train_dataset = MFTRCADSegDataset(
        root_dir=r'E:\CHB\python_project\SFRGNN\MFTRCAD',
        split='train',
        nums_data=16000,
        center_and_scale=False,
        normalize=True,
        random_rotate=True,
        num_threads=8
    )

    test_dataset = MFTRCADSegDataset(
        root_dir=r'E:\CHB\python_project\SFRGNN\MFTRCAD',
        split='test',
        nums_data=3400,
        center_and_scale=False,
        normalize=True,
        random_rotate=True,
        num_threads=8
    )

    test_loader = test_dataset.get_dataloader(batch_size=8, shuffle=True, pin_memory=True)

    for data in test_loader:
        if data is not None:
            print("Batch data:")
            print(data)

            # 打印语义分割标签
            if data["graph"] is not None and 'seg_y' in data["graph"].ndata:
                print("Segmentation Labels:", data["graph"].ndata['seg_y'])
        break  # 只打印第一个批次
