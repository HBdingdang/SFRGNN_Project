import pickle
import h5py
import numpy as np
import os

def pkl_to_h5py(pkl_path, h5_path):
    # 读取 PKL 文件
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    # 创建 HDF5 文件
    with h5py.File(h5_path, 'w') as h5_file:
        def recursively_save_dict_contents_to_group(h5file, path, dic):
            for key, item in dic.items():
                if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                    h5file[path + key] = item
                elif isinstance(item, dict):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
                else:
                    raise ValueError(f'Cannot save {type(item)} type')

        recursively_save_dict_contents_to_group(h5_file, '/', data)

# 示例使用
pkl_path = '/mnt/data.pkl'  # 替换为你的 PKL 文件路径
h5_path = 'path/to/your/data.h5'    # 替换为你希望保存 HDF5 文件的路径

pkl_to_h5py(pkl_path, h5_path)
