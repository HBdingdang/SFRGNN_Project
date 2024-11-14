
from tqdm import tqdm

import torch
from torch import nn

import numpy as np

from torchmetrics.classification import (
    MulticlassAccuracy, 
    BinaryAccuracy, 
    BinaryF1Score, 
    BinaryJaccardIndex, 
    MulticlassJaccardIndex,
    BinaryAveragePrecision)

from dataloader.mfinstseg import MFInstSegDataset
from dataloader.mftrcad import MFTRCADSegDataset
from models.segmentors import SFRGNNSegmentor
from utils.misc import seed_torch



if __name__ == '__main__':
    # track hyperparameters and run metadata
    torch.set_float32_matmul_precision("high") # may be faster if GPU support TF32
    config={
        "edge_attr_dim": 12,
        "node_attr_dim": 10,
        "edge_attr_emb": 64, # recommend: 64
        "node_attr_emb": 64, # recommend: 64
        "edge_grid_dim": 0, 
        "node_grid_dim": 7,
        "edge_grid_emb": 0, 
        "node_grid_emb": 64, # recommend: 64
        "num_layers": 3, # recommend: 2
        "delta": 2,
        "mlp_ratio": 2,
        "drop": 0.25, 
        "drop_path": 0.25,
        "head_hidden_dim": 64,
        "conv_on_edge": False,
        "use_uv_gird": True,
        "use_edge_attr": True,
        "use_face_attr": True,

        "seed": 42,
        "device": 'cuda',
        "architecture": "SFRGraphEncoder",
        "dataset_type": "full",
        "dataset": "/path/dataset",#

        "epochs": 100,
        "lr": 1e-2,
        "weight_decay": 1e-2,
        "batch_size": 256,
        "ema_decay_per_epoch": 1. / 2.,
        "seg_a": 1.,
        "inst_a": 1.,
        "bottom_a": 1.,
        }

    seed_torch(config['seed'])
    device = config['device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("GPU is enabled.")
    else:
        print("GPU is not enabled. Using CPU instead.")
    dataset = config['dataset']
    dataset_type = config['dataset_type']
    n_classes = 25

    model = SFRGNNSegmentor(num_classes=n_classes,
                            arch=config['architecture'],
                            edge_attr_dim=config['edge_attr_dim'], 
                            node_attr_dim=config['node_attr_dim'], 
                            edge_attr_emb=config['edge_attr_emb'], 
                            node_attr_emb=config['node_attr_emb'],
                            edge_grid_dim=config['edge_grid_dim'], 
                            node_grid_dim=config['node_grid_dim'], 
                            edge_grid_emb=config['edge_grid_emb'], 
                            node_grid_emb=config['node_grid_emb'], 
                            num_layers=config['num_layers'], 
                            delta=config['delta'], 
                            mlp_ratio=config['mlp_ratio'], 
                            drop=config['drop'], 
                            drop_path=config['drop_path'], 
                            head_hidden_dim=config['head_hidden_dim'],
                            conv_on_edge=config['conv_on_edge'],
                            use_uv_gird=config['use_uv_gird'],
                            use_edge_attr=config['use_edge_attr'],
                            use_face_attr=config['use_face_attr'],)
    model = model.to(device)

    model_param = torch.load("/path/weight", map_location=device)
    model.load_state_dict(model_param)


    test_dataset = MFTRCADSegDataset(root_dir=dataset,split='test',nums_data=3400,
        center_and_scale=False, normalize=True,random_rotate=True,
        num_threads=8
    )
    test_loader = test_dataset.get_dataloader(batch_size=config['batch_size'], pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()

    
    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)

    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)


    best_acc = 0.
    with torch.no_grad():
        print(f'------------- Now start testing ------------- ')
        model.eval()
        test_losses = []

        # 更新准确率和IOU的状态
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)

            seg_label = graphs.ndata["seg_y"]

            # Forward pass
            seg_pred = model(graphs)

            loss_seg = seg_loss(seg_pred, seg_label)

            loss = config['seg_a'] * loss_seg
            test_losses.append(loss.item())

            # 更新指标状态
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)  # 假设您有类似的IOU指标

        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()  # 计算最终的准确率
        mean_test_seg_iou = test_seg_iou.compute().item()  # 计算最终的IOU

        print(f'test_loss : {mean_test_loss}, \
              test_seg_acc: {mean_test_seg_acc}, \
              test_seg_iou: {mean_test_seg_iou}'
              )
