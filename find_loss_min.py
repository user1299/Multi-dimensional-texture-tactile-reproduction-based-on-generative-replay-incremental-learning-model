import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


directory_path = "/SATA2/DY/bimamba/datasave/"

min_mae = float('inf')
min_mae_file = None

csv_files = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

for file_path in tqdm(csv_files, desc="计算MAE进度"):
    try:
        data = pd.read_csv(file_path)
        if data.shape[1] >= 2:
            mae = mean_absolute_error(data.iloc[:, 0], data.iloc[:, 1])
            if mae < min_mae:
                min_mae = mae
                min_mae_file = file_path

    except Exception as e:
        print(f"无法处理文件 {file_path}: {e}")

if min_mae_file:
    print(f"\n最小 MAE 值: {min_mae}")
    print(f"对应的文件路径: {min_mae_file}")
else:
    print("未找到可处理的 CSV 文件。")
