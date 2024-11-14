import os


def find_unmatched_files(steps_folder, labels_folder):
    # 获取steps文件夹中的所有文件名（不包括扩展名）
    steps_files = set(os.path.splitext(file)[0] for file in os.listdir(steps_folder) if file.endswith('.step'))

    # 获取labels文件夹中的所有文件名（不包括扩展名）
    labels_files = set(os.path.splitext(file)[0] for file in os.listdir(labels_folder) if file.endswith('.json'))

    # 找到在labels文件夹中存在但在steps文件夹中不存在的文件
    unmatched_files = labels_files - steps_files

    return unmatched_files


# 指定文件夹路径
steps_folder = '/mnt/data'  # 替换为实际steps文件夹路径
labels_folder = '/mnt/data'  # 替换为实际labels文件夹路径

# 查找不匹配的文件
unmatched_files = find_unmatched_files(steps_folder, labels_folder)

# 打印不匹配的文件
if unmatched_files:
    print("以下文件在labels文件夹中存在但在steps文件夹中不存在:")
    for file in unmatched_files:
        print(file + '.json')
else:
    print("所有文件都匹配。")
