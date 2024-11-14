# 首先读取文件内容
file_path = "/mnt/data.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 计算分割的行数
total_lines = len(lines)
train_split = int(total_lines * 0.7)
val_split = train_split + int(total_lines * 0.15)

# 划分数据
train_data = lines[:train_split]
val_data = lines[train_split:val_split]
test_data = lines[val_split:]

# 写入新文件
train_path = "../xxx/train.txt"
val_path = "../xxx/val.txt"
test_path = "../xxx/test.txt"

with open(train_path, 'w', encoding='utf-8') as file:
    file.writelines(train_data)

with open(val_path, 'w', encoding='utf-8') as file:
    file.writelines(val_data)

with open(test_path, 'w', encoding='utf-8') as file:
    file.writelines(test_data)
