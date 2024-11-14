import pathlib
import json


import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader

from dataloader.base import BaseDataset
from utils.data_utils import load_one_graph, load_json_or_pkl, load_statistics, standardization, center_and_scale
import threading

from torch.utils.data import Sampler

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
    remap_dict = {1: 0, 12: 1, 14: 2, 6: 3, 0: 4, 23: 5, 24: 6}

    @staticmethod
    def num_classes(type='full'):
        return 25 if type == 'full' else 7

    @staticmethod
    def remap(logit):
        return MFInstSegAdaptiveDataset.remap_dict[logit]

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
                 dataset_type='full',
                 num_threads=0):
        super().__init__(transform, random_rotate)
        self.source_path = pathlib.Path(source_dir)
        self.target_path = pathlib.Path(target_dir)
        self.dataset_type = dataset_type
        self.normalize = normalize
        self.center_and_scale = center_and_scale
        assert split in ("train", "val", "test", "all")

        split_filelist_source = self._load_files(self.source_path, split, nums_data)
        split_filelist_target = self._load_files(self.target_path, split, nums_data)
        print(
            f"Loaded {len(split_filelist_source)} source files and {len(split_filelist_target)} target files for split {split}")
        self.graphs_source = self.load_graphs(self.source_path, graphs_source, graphs_target, split_filelist_source, True, num_threads)
        self.graphs_target = self.load_graphs(self.target_path, graphs_source, graphs_target, split_filelist_target, False, num_threads)


        self.data_source = self.graphs_source #加载了源域文件列表且含标签的数据
        self.data_target = self.graphs_target #加载了源域文件列表且含标签的数据
        self.data = self.data_source + self.data_target
        print(f"Done loading {len(self.data)} files for {split} split")

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

    # def load_graphs(self, file_path, graphs, split_file_list, num_threads=4):
    def load_graphs(self, file_path, source_graphs, target_graphs, split_file_list, has_labels, num_threads=4):
        self.data = []
        # if graphs:
        #     self.dataset = graphs
        # else:
        #     self.dataset = load_json_or_pkl(file_path.joinpath("aag").joinpath('graphs.json'))
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
        # 这一行代码使用列表推导式将数据集分割成多个子集。它的作用是从数据集中按 chunk_size 的大小逐步取出子集，直到取完所有的数据。
        chunks = [self.dataset[i:i+chunk_size] for i in range(0, len(self.dataset), chunk_size)]
        #对于每个 i，切片操作 self.dataset[i:i+chunk_size] 从数据集中取出从 i 开始、长度为 chunk_size 的一段数据，形成一个子集。

        threads = []
        results = [[] for _ in range(num_threads)]
        for i in range(num_threads):
            t = threading.Thread(target=lambda i: results[i].extend(
                self.process_chunk(
                    chunks[i], split_file_list,
                    # self.normalize, self.center_and_scale, stat, file_path)), args=(i,))
                    self.normalize, self.center_and_scale, stat, file_path, has_labels)), args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.data = [item for sublist in results for item in sublist]
        return self.data #文件列表中的所有数据，含域标签


    # def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path):
    def process_chunk(self, chunk, split_file_list, normalization_attribute, center_and_scale_grid, stat, file_path, has_labels):
        result = []
        for one_data in chunk:
            fn, data = one_data
            if fn in split_file_list:
                # one_graph = self.load_one_graph(fn, data)
                one_graph = self.load_one_graph(fn, data, has_labels) #return sample 一个图数据-对应一个文件数据，已经加载了label

                if one_graph is None:
                    continue
                if one_graph["graph"].edata["x"].size(0) == 0:
                    continue
                if normalization_attribute and stat:
                    one_graph = standardization(one_graph, stat)
                if center_and_scale_grid:
                    one_graph = center_and_scale(one_graph)
                result.append(one_graph) #将加载的文件列表中的数据加入res
        return result



    def load_one_graph(self, fn, data, has_labels):
        sample = load_one_graph(fn, data) # return   sample = {"graph": dgl_graph, "filename": fn}
        if sample is None:
            print(f"Failed to load graph for file: {fn}")
            return None

        num_faces = sample['graph'].num_nodes()
        if has_labels:
            label_file = self.source_path.joinpath("labels").joinpath(fn + ".json")
            if not label_file.exists():
                print(f"Label file not found: {label_file}")
                return None
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)
            _, labels = labels_data[0]
            seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']
            assert len(seg_label) == len(inst_label) and len(seg_label) == len(bottom_label)
            assert num_faces == len(seg_label)
            face_segmentaion_labels = np.zeros(num_faces)
            for idx, face_id in enumerate(range(num_faces)):
                index = seg_label[str(face_id)]
                face_segmentaion_labels[idx] = self.remap(index) if self.dataset_type == 'tiny' else index
            instance_label = np.array(inst_label, dtype=np.int32)
            bottom_segmentaion_labels = np.zeros(num_faces)
            for idx, face_id in enumerate(range(num_faces)):
                index = bottom_label[str(face_id)]
                bottom_segmentaion_labels[idx] = index
            sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentaion_labels).long()
            sample["inst_y"] = torch.tensor(instance_label).float()
            sample["graph"].ndata["bottom_y"] = torch.tensor(bottom_segmentaion_labels).float().reshape(-1, 1)
        else:
            label_file = self.target_path.joinpath("labels").joinpath(fn + ".json")
            if not label_file.exists():
                print(f"Label file not found: {label_file}")
                return None
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)
            _, labels = labels_data[0]
            seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']
            assert len(seg_label) == len(inst_label) and len(seg_label) == len(bottom_label)
            assert num_faces == len(seg_label)
            face_segmentaion_labels = np.zeros(num_faces)
            for idx, face_id in enumerate(range(num_faces)):
                index = seg_label[str(face_id)]
                face_segmentaion_labels[idx] = self.remap(index) if self.dataset_type == 'tiny' else index
            instance_label = np.array(inst_label, dtype=np.int32)
            bottom_segmentaion_labels = np.zeros(num_faces)
            for idx, face_id in enumerate(range(num_faces)):
                index = bottom_label[str(face_id)]
                bottom_segmentaion_labels[idx] = index
            sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentaion_labels).long()
            sample["inst_y"] = torch.tensor(instance_label).float()
            sample["graph"].ndata["bottom_y"] = torch.tensor(bottom_segmentaion_labels).float().reshape(-1, 1)
            # 确保所有图的边数据有相同的字段
        if "grid" not in sample["graph"].edata:
            num_edges = sample["graph"].number_of_edges()
            sample["graph"].edata["grid"] = torch.zeros((num_edges, 12, 5), dtype=torch.float32)

        sample['domain'] = 'source' if has_labels else 'target'
        sample["graph"].ndata['domain'] = torch.tensor([1 if has_labels else 0] * num_faces, dtype=torch.long)
        return sample


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        batch_size = len(batch)
        half_size = batch_size // 2

        # 前半部分来自源域，后半部分来自目标域
        source_batch = batch[:half_size]
        target_batch = batch[half_size:]

        final_batch = source_batch + target_batch

        batched_graph = dgl.batch([sample["graph"] for sample in final_batch])
        inst_labels = self.pack_pad_2D_adj(final_batch)
        batched_filenames = [sample["filename"] for sample in final_batch]

        return {"graph": batched_graph,
                "inst_labels": inst_labels,
                "filename": batched_filenames}




    def pack_pad_2D_adj(self, batch):
        max_nodes = max([sample["graph"].num_nodes() for sample in batch])
        batch_size = len(batch)
        inst_labels = torch.zeros((batch_size, max_nodes, max_nodes))
        for i, sample in enumerate(batch):
            num_nodes = sample["graph"].num_nodes()
            inst_labels[i, :num_nodes, :num_nodes] = sample["inst_y"]
        return inst_labels

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
    source_dir = '/mnt/data/CHB/AAGNet-main/MFInstSeg'
    target_dir = '/mnt/data/CHB/AAGNet-main/dataset/data_7w'
    train_dataset = MFInstSegAdaptiveDataset(source_dir=source_dir, target_dir=target_dir, split='train',
                                             center_and_scale=True, normalize=False, num_threads=8, nums_data=100)
    # 打印数据集大小
    print(f"Total dataset size: {len(train_dataset)}")
    source_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'source'])
    target_data_size = len([sample for sample in train_dataset.data if sample['domain'] == 'target'])
    print(f"Source data size: {source_data_size}, Target data size: {target_data_size}")

    train_loader = train_dataset.get_dataloader(batch_size=8, pin_memory=True)

    # for data in train_loader:
    #     if data is not None:
    #         print(data)
    for data in train_loader:
        if data is not None:
            print("Batch data:")
            print(data)

            # 提取并打印标签
            inst_label = data["inst_labels"]
            seg_label = data["graph"].ndata["seg_y"]
            bottom_label = data["graph"].ndata["bottom_y"]

            print("Instance labels:", inst_label)
            print("Segmentation labels:", seg_label)
            print("Bottom labels:", bottom_label)
            break  # 只打印第一个批次的数据，之后跳出循环
