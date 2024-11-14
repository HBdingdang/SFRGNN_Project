import os


def list_files(directory, output_file):
    # 获取指定目录下的所有文件和子目录，只查看当前目录层级
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 对文件名进行排序
    filenames.sort()

    # 打开输出文件准备写入
    with open(output_file, 'w') as file:
        for filename in filenames:
            # 移除文件扩展名并写入文件，每个文件名占一行
            file.write(os.path.splitext(filename)[0] + '\n')


# 使用示例
directory_path = '/mnt/data'  # 你需要指定的目录路径
output_file_path = '/mnt/data'  # 输出文件的路径
list_files(directory_path, output_file_path)
