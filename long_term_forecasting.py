import pandas as pd
import os
import time
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import BiMamba4TS
import torch
from utils.loss import get_loss
from tqdm import tqdm
import csv

models_dict = {
    'BiMamba4TS': BiMamba4TS
}

optimizer_catagory = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}

loss_funcs = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
    "huber": torch.nn.HuberLoss(reduction='mean', delta=1.0)
}


class LTF_Trainer:
    def __init__(self, args, task, setting, corr=None) -> None:
        self.args = args
        self.setting = setting
        self.r_path = args.results
        self.c_path = args.checkpoints
        self.p_path = args.predictions
        self.device = torch.device(args.device)

        # 确保路径存在
        os.makedirs(self.c_path, exist_ok=True)
        os.makedirs(self.r_path, exist_ok=True)
        os.makedirs(self.p_path, exist_ok=True)

        print(f"Checkpoints path: {self.c_path}")
        print(f"Results path: {self.r_path}")
        print(f"Predictions path: {self.p_path}")

        if hasattr(args, 'SRA') and args.SRA:
            model = models_dict[self.args.model].Model(self.args, corr=corr).to(self.device)
        else:
            model = models_dict[self.args.model].Model(self.args).to(self.device)

        if args.use_multi_gpu:
            self.model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        else:
            self.model = model

    def train(self, data):
        print('>>>>>> start training : {} >>>>>>'.format(self.setting))
    
        train_loader = data.get('combined_train_loader', None)  # 训练集加载器
        train_steps = len(train_loader)
    
        optimizer = optimizer_catagory[self.args.opt](self.model.parameters(), lr=self.args.learning_rate)
        loss_func = loss_funcs[self.args.loss]
    
        train_loss_list, epoch_list = [], []
    
        for epoch in range(self.args.train_epochs):
            epoch_list.append(epoch + 1)
            epoch_time = time.time()  # 记录训练开始时间
            train_loss = []
            batch_data_list = []
            self.model.train()
    
            # 创建训练进度条
            with tqdm(total=train_steps, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}", unit='batch',
                      leave=True) as progress_bar:
                # 训练过程
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    optimizer.zero_grad()
    
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y.float().to(self.args.device)
    
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.label_len:, :]).float()
                    dec_inp = torch.cat([batch_x, dec_inp], dim=1).float()
    
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, _ = outputs
                    f_dim = -1 if self.args.task == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
    
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)
                    loss = loss_func(outputs, batch_y)
    
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
    
                    # 保存第7-10个batch的数据（索引6-9）
                    if 6 <= i <= 9:  # 包含第7个（i=6）到第10个（i=9）batch
                        # 转换为numpy并展平（保持原有维度处理逻辑）
                        current_outputs = outputs.cpu().detach().numpy()
                        current_batch_y = batch_y.cpu().detach().numpy()
                        
                        # 为每个数据添加batch标识（方便后续区分）
                        batch_id = i + 1  # 转换为1-based编号（7-10）
                        
                        # 处理多维度数据（展平为二维：样本+时间步）
                        if current_outputs.ndim == 3:
                            # 假设shape为(batch_size, seq_len, features)，取第一个特征或展平特征
                            current_outputs = current_outputs.reshape(-1, current_outputs.shape[-1])
                            current_batch_y = current_batch_y.reshape(-1, current_batch_y.shape[-1])
                        
                        # 构建DataFrame并添加batch_id
                        df_batch = pd.DataFrame({
                            'batch_id': [batch_id] * len(current_outputs),  # 标记所属batch
                            'outputs': current_outputs.flatten(),
                            'batch_y': current_batch_y.flatten()
                        })
                        batch_data_list.append(df_batch)
    
                    # 更新训练进度条
                    progress_bar.set_postfix({'train_loss': np.mean(train_loss)})
                    progress_bar.update(1)
    
            epoch_duration = time.time() - epoch_time  # 计算训练耗时
            print(f"Epoch: {epoch + 1}, Train Loss: {np.mean(train_loss):.4f} | Training Time: {epoch_duration:.2f} seconds")
    
        # 保存损失结果
        loss_save_path = os.path.join(self.r_path, 'loss.csv')
        loss_data = pd.DataFrame({
            'epoch': epoch_list,
            'train_loss': train_loss_list
        })
        loss_data.to_csv(loss_save_path, index=False)
    
        # 绘制损失曲线
        _, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(epoch_list, train_loss_list, 'r', marker='x', label='train_loss')
        ax.legend()
        ax.set_title('Loss Function Value')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(self.r_path + '/loss.png', dpi=100)
        plt.clf()
    
    def validate(self, vali_loader, loss_func, progress_bar):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.label_len:, :]).float()
                dec_inp = torch.cat([batch_x, dec_inp], dim=1).float()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, _ = outputs

                f_dim = -1 if self.args.task == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                loss = loss_func(outputs, batch_y)
                total_loss.append(loss.item())

                # 更新验证进度条
                progress_bar.set_postfix({'vali_loss': np.mean(total_loss)})
                progress_bar.update(1)

        total_loss = np.average(total_loss)
        return total_loss