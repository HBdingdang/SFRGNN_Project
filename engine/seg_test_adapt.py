
from tqdm import tqdm

from torch import nn


from dataloader.mfinstseg import MFInstSegDataset

from models.inst_segmentors_byChb_adapt_SegTask import SFRGNNSegmentor

from utils.misc import seed_torch

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

import torch

def compute_per_class_accuracy(predictions, labels, num_classes):
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # Iterate through all the predictions and labels
    for i in range(predictions.size(0)):  # For each instance in the batch
        label = labels[i]
        prediction = torch.argmax(predictions[i])  # Get the predicted class

        # Update count for the true label and predicted label
        class_total[label] += 1
        if prediction == label:
            class_correct[label] += 1

    # Calculate accuracy for each class
    class_accuracy = [0] * num_classes
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0  # If no samples for that class

    return class_accuracy

if __name__ == '__main__':
    # track hyperparameters and run metadata
    torch.set_float32_matmul_precision("high")  # may be faster if GPU support TF32
    config = {
        "edge_attr_dim": 12,
        "node_attr_dim": 10,
        "edge_attr_emb": 64,  # recommend: 64
        "node_attr_emb": 64,  # recommend: 64
        "edge_grid_dim": 0,
        "node_grid_dim": 7,
        "edge_grid_emb": 0,
        "node_grid_emb": 64,  # recommend: 64
        "num_layers": 3,  # recommend: 3
        "delta": 2,  # obsolete
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
        "dataset": "xxxxxxxxxxxxxxxxx",
        "adaptation_method": "madann",  # madann, mmd, tca

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

    n_classes = 25
    device = config['device']

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
                            use_face_attr=config['use_face_attr'],
                            adaptation_method=config['adaptation_method'])
    model = model.to(device)

    model_param = torch.load("xxxxxxxxxxxxxxxxx",map_location=device)
    model.load_state_dict(model_param)
    dataset = config['dataset']
    test_dataset = MFInstSegDataset(root_dir=dataset, split='test', nums_data=3400,
                                     center_and_scale=False, normalize=True,
                                     num_threads=8)
    test_loader = test_dataset.get_dataloader(batch_size=config['batch_size'], pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()

    # Metrics initialization
    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    best_acc = 0.
    with torch.no_grad():
        print(f'------------- Now start testing -------------')
        model.eval()
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]

            # Forward pass
            seg_pred, _ = model(graphs)

            # Compute loss
            loss_seg = seg_loss(seg_pred, seg_label)
            loss = loss_seg
            test_losses.append(loss.item())

            # Update metrics
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)

        # Compute overall accuracy
        overall_accuracy = test_seg_acc.compute()
        overall_iou = test_seg_iou.compute()

        # Compute per-class accuracy
        per_class_accuracy = compute_per_class_accuracy(seg_pred, seg_label, n_classes)

        # Compute average per-class accuracy
        average_per_class_accuracy = sum(per_class_accuracy) / len(per_class_accuracy)

        # Print results
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall IoU: {overall_iou:.4f}")
        print(f"Per-Class Accuracy: {per_class_accuracy}")
        print(f"Average Per-Class Accuracy: {average_per_class_accuracy:.4f}")

        # Reset metrics for the next evaluation
        test_seg_acc.reset()
        test_seg_iou.reset()
