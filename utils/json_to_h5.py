import json
import h5py
import numpy as np
import torch


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_to_hdf5(data, file_path):
    with h5py.File(file_path, 'w') as f:
        for i, graph_data in enumerate(data):
            fn = graph_data[0]  # 第一个元素是文件名
            graph_data = graph_data[1]  # 第二个元素是图数据
            grp = f.create_group(str(i))
            grp.attrs['filename'] = fn

            # 添加图的边和节点数
            grp.create_dataset('edges', data=np.array(graph_data['graph']['edges']))
            grp.create_dataset('num_nodes', data=graph_data['graph']['num_nodes'])

            # 添加节点属性
            node_attributes = np.array(graph_data['graph_face_attr'])
            grp.create_dataset('node_features_x', data=node_attributes)

            # 添加节点网格属性
            if len(graph_data['graph_face_grid']) > 0:
                node_grid_attributes = np.array(graph_data['graph_face_grid'])
                grp.create_dataset('node_grid', data=node_grid_attributes)

            # 添加边属性
            edge_attributes = np.array(graph_data['graph_edge_attr'])
            grp.create_dataset('edge_features_x', data=edge_attributes)

            # 添加边网格属性
            if len(graph_data['graph_edge_grid']) > 0:
                edge_grid_attributes = np.array(graph_data['graph_edge_grid'])
                grp.create_dataset('edge_grid', data=edge_grid_attributes)

def convert_json_to_h5(json_file, h5_file):
    data = load_json(json_file)
    save_to_hdf5(data, h5_file)


# 示例使用
json_file_path = '/mnt/data'
h5_file_path = '/mnt/data'

convert_json_to_h5(json_file_path, h5_file_path)
