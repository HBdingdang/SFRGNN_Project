import json
import pickle


def json_to_pkl(json_file_path, pkl_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 将数据写入 PKL 文件
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


# 示例使用
json_file_path = '/mnt/data'  # 替换为你的 JSON 文件路径
pkl_file_path = '/mnt/data'  # 替换为你希望保存的 PKL 文件路径

json_to_pkl(json_file_path, pkl_file_path)
