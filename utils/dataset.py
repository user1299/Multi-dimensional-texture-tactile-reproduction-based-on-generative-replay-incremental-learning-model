import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_Forecasting_Solar(Dataset):
    def __init__(self, raw_data, border1s, border2s, flag,
                 size, task, target, scale, device, pred_len, image_feature, step=50):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.step = step
        assert flag in ['train', 'test', 'val']
        self.set_type = 0 if flag == 'train' else 1 if flag == 'val' else 2
        self.target = target
        self.task = task
        self.scale = scale
        self.device = device
        self.pred_len = pred_len
        self.image_feature = image_feature  # 存储预先计算的图像特征
        self.__read_data__(raw_data, border1s, border2s)

    # 数据读取与预处理
    def __read_data__(self, df_raw, border1s, border2s):
        # 使用 StandardScaler 来标准化数据，并根据 set_type 设置数据边界
        self.scaler = StandardScaler()
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 根据任务类型选择数据
        if self.task == 'M' or self.task == 'MS':
            df_data = df_raw
        elif self.task == 'S':
            df_data = df_raw[[self.target]]
        # 标准化数据
        if self.scale:
            if self.set_type == 3:
                self.scaler.fit(df_data.values)
            else:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 设置数据集 x 和 y
        if self.set_type == 3:  # 预测集
            self.data_x = data[border1:border2 - self.pred_len]
            self.data_y = data[border1:border2]
        else:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    '''
    def __getitem__(self, index):
        index = index
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 提取输入序列 seq_x 和目标序列 seq_y
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 使用预先计算的图像特征
        image_feature = self.image_feature  # 直接使用存储的图像特征

        # 将图像特征拼接到输入序列
        seq_x_first_part = seq_x[:, :-1]
        seq_x_last_column = seq_x[:, -1:]
        seq_x = np.hstack((seq_x_first_part, image_feature.reshape(-1, 1), seq_x_last_column))

        # 将零列拼接到目标序列
        zero_column = np.zeros((seq_y.shape[0], 1))
        seq_y_first_part = seq_y[:, :-1]
        seq_y_last_column = seq_y[:, -1:]
        seq_y = np.hstack((seq_y_first_part, zero_column, seq_y_last_column))

        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    # 确定数据集的长度
    def __len__(self):
        if self.set_type == 3:
            return len(self.data_x) - self.seq_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1


'''

    def __getitem__(self, index):
        # 确定序列的起始和结束位置
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 提取输入序列 seq_x 和目标序列 seq_y
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 使用预先计算的图像特征
        image_feature = self.image_feature  # 直接使用存储的图像特征

        # 将图像特征拼接到输入序列
        seq_x_first_part = seq_x[:, :-1]  # 除最后一列外的部分
        seq_x_last_column = seq_x[:, -1:]  # 最后一列
        seq_x = np.hstack((seq_x_first_part, image_feature.reshape(-1, 1), seq_x_last_column))  # 拼接图像特征

        # 将零列拼接到目标序列
        zero_column = np.zeros((seq_y.shape[0], 1))
        seq_y_first_part = seq_y[:, :-1]  # 目标序列除最后一列
        seq_y_last_column = seq_y[:, -1:]  # 目标序列最后一列
        seq_y = np.hstack((seq_y_first_part, zero_column, seq_y_last_column))  # 拼接零列

        # 创建标记序列
        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __iter__(self):
        self.index = 0  # 每次迭代开始时重置index
        return self

    def __next__(self):
        if self.index + self.seq_len + self.pred_len <= len(self.data_x):
            result = self.__getitem__(self.index)
            self.index += self.step  # 调整步长以控制采样的稀疏程度
            return result
        else:
            raise StopIteration

    def __len__(self):
        """
        返回数据集的长度
        """
        if self.set_type == 3:
            return (len(self.data_x) - self.seq_len + 1) // self.step  # 每次跳过一个，长度减少一半
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.step  # 每次跳过一个，长度减少一半
