import os
import numpy as np
import pandas as pd
from image import image
from torch.utils.data import DataLoader, ConcatDataset
import warnings
from utils.dataset import Dataset_Forecasting_Solar

warnings.filterwarnings('ignore')


# 根据数据集的长度、输入序列长度和预测长度划分训练、验证和测试集
def construct_borders(length: int, seq_len: int, pred_len: int):
    num_train = int(length * 0.7)
    num_vali = int(length * 0.2)
    num_test = length - num_train - num_vali
    # 每个数据集起始位置
    # border1s = [0, num_train - seq_len, length - num_test - seq_len, length - seq_len - pred_len]
    border1s = [0, num_train-4200, length - num_test - seq_len, length - seq_len - pred_len]
    # 结束位置
    # border2s = [num_train, num_train + num_vali, length, length]
    border2s = [num_train, num_train+4000, length, length]
    return border1s, border2s


def load_data(args):
    data_type_param = {
        'train': {'shuffle': False, 'drop_last': True, 'batch_size': args.batch_size},
        'val': {'shuffle': True, 'drop_last': True, 'batch_size': args.batch_size},
        'test': {'shuffle': False, 'drop_last': False, 'batch_size': args.batch_size},
    }

    csv_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.csv')]
    data = {}
    all_train_datasets = []  # 用于存储所有CSV文件的训练集
    all_val_datasets = []    # 用于存储所有CSV文件的验证集
    all_test_datasets = []   # 用于存储所有CSV文件的测试集
    correlation_matrix = None

    total_lengths = {
        'train': 0,
        'val': 0,
        'test': 0,
    }

    for i, csv_file in enumerate(csv_files):
        csv_path = os.path.join(args.dataset_path, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        image_file = f"{base_name}_square.png"
        image_path = os.path.join(args.images_dir, image_file)
        image_feature = image(image_path, args.device, args.seq_len)

        # 读取 CSV 文件
        df_raw = pd.read_csv(csv_path)

        # 选择列：去除目标列，限制特征列数
        cols = list(df_raw.columns)
        if args.target not in cols:
            raise ValueError(f"Target column '{args.target}' not found in CSV file '{csv_file}'.")
        cols.remove(args.target)
        cols = cols[0:args.enc_in - 1]
        df_raw = df_raw[cols + [args.target]]

        # 获取每个CSV文件的数据集边界
        border1s, border2s = construct_borders(len(df_raw), args.seq_len, args.pred_len)

        # 为每个CSV文件分别创建训练、验证和测试集的DataLoader
        for category, (border1, border2) in zip(['train', 'val', 'test'], zip(border1s, border2s)):
            dataset = Dataset_Forecasting_Solar(
                raw_data=df_raw,
                border1s=border1s,
                border2s=border2s,
                flag=category,
                size=[args.seq_len, args.label_len, args.pred_len],
                task=args.task,
                target=args.target,
                scale=True,
                device=args.device,
                pred_len=args.pred_len,
                image_feature=image_feature
            )

            # 存储每个文件的DataLoader
            loader = DataLoader(dataset, **data_type_param[category])

            # 如果是训练集，将其数据加入到all_train_datasets
            if category == 'train':
                all_train_datasets.append(dataset)
            elif category == 'val':
                all_val_datasets.append(dataset)  # 添加到验证集列表
            elif category == 'test':
                all_test_datasets.append(dataset)  # 添加到测试集列表

            # 使用文件索引区分不同文件的数据
            data[f"{category}_loader_{i}"] = loader

            # 更新每个类别的总长度
            total_lengths[category] += len(dataset)

    # 合并所有训练集、验证集和测试集的数据集，生成统一的DataLoader
    if all_train_datasets:
        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_train_loader = DataLoader(combined_train_dataset, **data_type_param['train'])
        data['combined_train_loader'] = combined_train_loader  # 存储合并后的训练数据

    if all_val_datasets:
        combined_val_dataset = ConcatDataset(all_val_datasets)
        combined_val_loader = DataLoader(combined_val_dataset, **data_type_param['val'])
        data['combined_val_loader'] = combined_val_loader  # 存储合并后的验证数据

    if all_test_datasets:
        combined_test_dataset = ConcatDataset(all_test_datasets)
        combined_test_loader = DataLoader(combined_test_dataset, **data_type_param['test'])
        data['combined_test_loader'] = combined_test_loader  # 存储合并后的测试数据

    # 打印每个类别的总长度
    for category, length in total_lengths.items():
        print(f"Total sample size of {category}: {length}")

    # 计算特征相关性矩阵
    if hasattr(args, 'SRA') and args.SRA:
        df_train = df_raw[cols + [args.target]][border1s[0]:border2s[0]]
        correlation_matrix = df_train.corr(method=args.relation)

    return data, np.array(correlation_matrix) if correlation_matrix is not None else None

