# utf-8
import json
import torch

if __name__ == '__main__':
    with open('/mnt/data/CHB/AAGNet-main/dataset/data_7w/aag/graphs.json', 'r') as fp:
        f = json.load(fp)
    torch.save(f, '/mnt/data/CHB/AAGNet-main/dataset/data_7w/graphs.pkl')

