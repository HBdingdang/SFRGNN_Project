import pathlib
import json
import os

import torch
import dgl
import math
import numpy as np

from .base import BaseDataset
from utils.data_utils import load_one_graph



class MFCAD2Dataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 25
    
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
        Load the MFInstSeg Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            graphs (list, optional): List of graph data.
            split (str, optional): Data split to load. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
            nums_data (int, optional): Number of training examples to use. Defaults to -1 (all training examples will be used).
            transform (callable, optional): Transformation to apply to the data.
        """
        path = pathlib.Path(root_dir)
        self.path = path
        self.transform = transform
        self.random_rotate = random_rotate
        assert split in ("train", "val", "test")

        filelist = {}
        data = np.loadtxt(str(path.joinpath(f"{split}.txt")), dtype=str)
        filelist[split] = data

        if split == "train":
            split_filelist = filelist["train"][:nums_data]
        elif split == "val":
            split_filelist = filelist["val"][:nums_data]
        else:
            split_filelist = filelist["test"][:nums_data]

        self.random_rotate = random_rotate

        # Load graphs
        print(f"Loading {split} data...")
        split_filelist = set(split_filelist)
        graph_path = path.joinpath("aag")
        self.load_graphs(graph_path, graphs, split_filelist, center_and_scale, normalize)
        print("Done loading {} files".format(len(self.data)))

    def _collate(self, batch):
        """
        Collate a batch of data samples together into a single batch.

        Args:
            batch (List[dict]): List of data samples.

        Returns:
            dict: Batched data.
        """
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph,
                "filename": batched_filenames}
    
    def load_one_graph(self, fn, data):
        """
        Load the data for a single file.

        Args:
            fn (str): Filename.
            data (dict): Data for the file.

        Returns:
            dict: Data for the file.
        """
        # Load the graph using base class method
        sample = load_one_graph(fn, data)
        # Additionally load the label and store it as node data
        label_file = self.path.joinpath("labels").joinpath(fn + ".json")
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        labels_data = np.array(labels_data, dtype=np.int32)
        sample["graph"].ndata["seg_y"] = torch.tensor(labels_data).long()
        return sample



    