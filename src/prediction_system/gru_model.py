import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DrivingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层，用于输出预测值
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 初始化隐藏状态，如果没有提供的话
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU层前向传播
        out, h_out = self.gru(x, h)  # out shape: (batch_size, sequence_length, hidden_size)
        
        # 我们只需要最后一个时间步的输出
        out = self.fc(out[:, -1, :])  # shape: (batch_size, output_size)
        
        return out, h_out

class FatiguePredictionSystem:
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, learning_rate=0.001, 
                 batch_size=32, sequence_length=60, prediction_horizon=20,
                 samples_per_hour=60):
        """
        初始化疲劳预测系统
        
        参数:
        - input_size: 输入特征维度（默认为5个特征）
        - hidden_size: GRU隐藏层大小
        - num_layers: GRU层数
        - learning_rate: 学习率
        - batch_size: 批次大小
        - sequence_length: 输入序列长度（1小时的数据点数量）
        - prediction_horizon: 预测时间范围（20分钟后）
        - samples_per_hour: 每小时的样本数量
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.samples_per_hour = samples_per_hour
        
        # 检查CUDA是否可用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        ).to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 用于数据预处理的scaler
        self.scaler = StandardScaler()
        
    def prepare_data_sliding_window(self, csv_file_path):
        """
        使用滑动窗口方法从CSV文件或目录准备数据
        
        参数:
        - csv_file_path: CSV文件或目录路径
        
        返回:
        - 处理后的数据帧和标准化器
        """
        # 检查是否是目录
        if os.path.isdir(csv_file_path):
            print(f"检测到目录: {csv_file_path}")
            # 获取目录下所有的CSV文件
            csv_files = [os.path.join(csv_file_path, file) 
                        for file in os.listdir(csv_file_path) 
                        if file.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"目录 {csv_file_path} 中没有找到CSV文件")
            
            print(f"找到 {len(csv_files)} 个CSV文件")
            
            # 读取并合并所有CSV文件
            dfs = []
            for file in tqdm(csv_files, desc="读取CSV文件"):
                try:
                    df = pd.read_csv(file)
                    
                    # 分析采样率
                    if 'Timestamp' in df.columns:
                        try:
                            # 转换时间戳列为datetime
                            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                            # 计算时间间隔（以秒为单位）
                            if len(df) > 1:
                                time_diffs = (df['Timestamp'].diff().iloc[1:].dt.total_seconds())
                                avg_interval = time_diffs.mean()
                                print(f"文件 {os.path.basename(file)} 的平均采样间隔: {avg_interval:.2f} 秒")
                                # 估计每小时样本数
                                samples_per_hour_est = int(3600 / avg_interval)
                                print(f"估计每小时样本数: {samples_per_hour_est}")
                                
                                # 更新samples_per_hour（如果不同于当前值）
                                if abs(self.samples_per_hour - samples_per_hour_est) > 100:  # 允许一些误差
                                    print(f"更新每小时样本数从 {self.samples_per_hour} 到 {samples_per_hour_est}")
                                    self.samples_per_hour = samples_per_hour_est
                        except Exception as e:
                            print(f"分析时间戳时出错: {e}")
                    
                    # 选择需要的列
                    selected_features = ['Blink Count', 'Blink Rate(per minute)', 'Yawn Count', 
                                        'Yawn Rate(per hour)', 'Fatigue Score']
                    if all(col in df.columns for col in selected_features):
                        df = df[selected_features]
                        # 处理缺失值
                        df = df.ffill()
                        dfs.append(df)
                    else:
                        missing_cols = [col for col in selected_features if col not in df.columns]
                        print(f"文件 {file} 缺少列: {missing_cols}")
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
            
            if not dfs:
                raise ValueError("没有成功读取任何CSV文件")
            
            # 合并所有数据帧
            df = pd.concat(dfs, ignore_index=True)
            print(f"合并后的数据大小: {df.shape}")
        else:
            # 直接读取单个CSV文件
            print(f"读取单个CSV文件: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            # 分析采样率
            if 'Timestamp' in df.columns:
                try:
                    # 转换时间戳列为datetime
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    # 计算时间间隔（以秒为单位）
                    if len(df) > 1:
                        time_diffs = (df['Timestamp'].diff().iloc[1:].dt.total_seconds())
                        avg_interval = time_diffs.mean()
                        print(f"平均采样间隔: {avg_interval:.2f} 秒")
                        # 估计每小时样本数
                        samples_per_hour_est = int(3600 / avg_interval)
                        print(f"估计每小时样本数: {samples_per_hour_est}")
                        
                        # 更新samples_per_hour（如果不同于当前值）
                        if abs(self.samples_per_hour - samples_per_hour_est) > 100:  # 允许一些误差
                            print(f"更新每小时样本数从 {self.samples_per_hour} 到 {samples_per_hour_est}")
                            self.samples_per_hour = samples_per_hour_est
                except Exception as e:
                    print(f"分析时间戳时出错: {e}")
            
            # 选择需要的列
            selected_features = ['Blink Count', 'Blink Rate(per minute)', 'Yawn Count', 
                                'Yawn Rate(per hour)', 'Fatigue Score']
            df = df[selected_features]
            
            # 处理缺失值
            df = df.ffill()
        
        # 创建特征和标签数据
        all_data = df.values
        
        # 标准化特征数据
        self.scaler.fit(all_data)
        scaled_data = self.scaler.transform(all_data)
        
        # 将标准化后的数据转换回DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        
        return scaled_df
    
    def create_hour_windows(self, df, hour_windows=None, min_sequence_length=5, step_size=None):
        """
        创建1小时窗口的数据集
        
        参数:
        - df: 标准化后的数据帧
        - hour_windows: 指定要使用的小时窗口列表 (可选，默认使用所有可能的窗口)
        - min_sequence_length: 每个窗口内最小序列长度
        - step_size: 窗口滑动步长，默认为samples_per_hour（即不重叠）
        
        返回:
        - 小时窗口数据列表
        """
        # 总样本数
        total_samples = len(df)
        print(f"总样本数: {total_samples}")
        
        # 检查是否有足够的数据来创建一个完整的输入序列+预测间隔
        required_min_samples = self.sequence_length + self.prediction_horizon
        print(f"创建一个完整预测所需的最小样本数: {required_min_samples} (序列长度: {self.sequence_length} + 预测间隔: {self.prediction_horizon})")
        
        if total_samples < required_min_samples:
            print(f"警告: 总样本数 {total_samples} 小于所需的最小样本数 {required_min_samples}")
            return []
        
        # 如果未指定步长，使用默认值
        if step_size is None:
            step_size = self.samples_per_hour // 2  # 默认使用半个窗口作为步长，创建50%重叠的窗口
            print(f"使用默认步长: {step_size} (50%重叠)")
        
        # 计算可以创建的小时窗口数量 (考虑步长)
        max_windows = (total_samples - self.samples_per_hour) // step_size + 1
        if max_windows < 0:
            max_windows = 0
        
        print(f"可以创建的最大窗口数: {max_windows}, 每个窗口大小: {self.samples_per_hour}, 步长: {step_size}")
        print(f"序列长度: {self.sequence_length}, 预测间隔: {self.prediction_horizon}")
        
        if max_windows <= 0:
            print(f"警告: 没有足够的样本来创建完整的小时窗口. 总样本数: {total_samples}, 需要: {self.samples_per_hour}")
            # 尝试创建一个较小的窗口
            print(f"尝试创建一个包含所有数据的窗口...")
            
            # 计算可以创建的序列数
            available_sequences = total_samples - self.sequence_length - self.prediction_horizon + 1
            print(f"可创建的序列数: {available_sequences}")
            
            if available_sequences > 0:
                X, y = [], []
                
                # 创建尽可能多的序列
                for i in range(0, available_sequences):
                    # 当前序列
                    if i + self.sequence_length > len(df):
                        print(f"警告: 索引超出范围 - i={i}, sequence_length={self.sequence_length}, df长度={len(df)}")
                        continue
                        
                    seq = df.iloc[i:i+self.sequence_length].values
                    
                    # 预测目标
                    target_idx = i + self.sequence_length + self.prediction_horizon - 1
                    if target_idx >= len(df):
                        print(f"警告: 目标索引超出范围 - target_idx={target_idx}, df长度={len(df)}")
                        continue
                        
                    target = df.iloc[target_idx]['Fatigue Score']
                    X.append(seq)
                    y.append(target)
                
                print(f"创建的序列数: {len(X)}")
                if len(X) >= min_sequence_length:
                    X = np.array(X)
                    y = np.array(y).reshape(-1, 1)
                    
                    # 转换为PyTorch张量
                    X = torch.FloatTensor(X)
                    y = torch.FloatTensor(y)
                    
                    print(f"创建了一个包含 {len(X)} 个序列的窗口")
                    return [(X, y)]
                else:
                    print(f"序列数 {len(X)} 小于最小要求 {min_sequence_length}")
            else:
                print(f"没有足够的数据来创建序列: available_sequences={available_sequences}")
            return []
        
        # 如果没有指定窗口，则使用所有可能的窗口
        if hour_windows is None:
            hour_windows = list(range(max_windows))
        
        window_data = []
        
        for window_idx in hour_windows:
            start_idx = window_idx * step_size
            end_idx = min(start_idx + self.samples_per_hour, total_samples)  # 确保不超出数据范围
            
            # 计算当前窗口可以创建的序列数量
            window_available_sequences = end_idx - start_idx - self.sequence_length + 1
            window_available_sequences = max(0, window_available_sequences)  # 确保非负
            
            print(f"窗口 {window_idx}: 起始索引: {start_idx}, 结束索引: {end_idx}, 可创建序列数: {window_available_sequences}")
            
            # 确保我们有足够的数据来预测
            max_target_idx = start_idx + self.sequence_length + self.prediction_horizon - 1
            if max_target_idx >= total_samples:
                print(f"窗口 {window_idx}: 目标索引 {max_target_idx} 超出数据范围 {total_samples}，跳过")
                continue
                
            X, y = [], []
            
            # 为这个小时窗口创建序列
            valid_sequences = 0
            for i in range(start_idx, end_idx - self.sequence_length + 1):
                # 当前序列
                if i + self.sequence_length > len(df):
                    continue
                    
                seq = df.iloc[i:i+self.sequence_length].values
                
                # 预测目标 (20分钟后的疲劳分数)
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                if target_idx >= total_samples:
                    continue
                    
                target = df.iloc[target_idx]['Fatigue Score']
                X.append(seq)
                y.append(target)
                valid_sequences += 1
            
            print(f"窗口 {window_idx}: 有效序列数: {valid_sequences}")
            
            if len(X) >= min_sequence_length:  # 确保我们有足够的序列
                X = np.array(X)
                y = np.array(y).reshape(-1, 1)
                
                # 转换为PyTorch张量
                X = torch.FloatTensor(X)
                y = torch.FloatTensor(y)
                
                print(f"窗口 {window_idx}: 成功创建 {len(X)} 个序列")
                window_data.append((X, y))
            else:
                print(f"窗口 {window_idx}: 序列数 {len(X)} 小于最小要求 {min_sequence_length}，跳过")
        
        print(f"成功创建了 {len(window_data)} 个有效窗口")
        return window_data
    
    def train_with_sliding_window(self, csv_file_path, num_epochs=2, window_epochs=1,
                                  validation_ratio=0.2, patience=3, min_samples_per_window=10, window_step_size=None):
        """
        使用滑动窗口方法训练模型
        
        参数:
        - csv_file_path: CSV文件或目录路径
        - num_epochs: 整个数据集的训练轮次
        - window_epochs: 每个窗口的训练轮次
        - validation_ratio: 验证集比例
        - patience: 早停的耐心值
        - min_samples_per_window: 每个窗口的最小样本数
        - window_step_size: 窗口滑动步长，默认为半个窗口大小（50%重叠）
        
        返回:
        - 训练历史记录
        """
        # 准备数据
        scaled_df = self.prepare_data_sliding_window(csv_file_path)
        
        # 创建小时窗口数据
        window_data = self.create_hour_windows(scaled_df, min_sequence_length=min_samples_per_window, step_size=window_step_size)
        
        if not window_data:
            print("没有足够的数据来创建窗口！")
            return None
        
        print(f"创建了 {len(window_data)} 个小时窗口")
        
        # 用于记录训练过程
        train_history = {
            'train_losses': [],
            'val_losses': []
        }
        
        # 用于早停的变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 多轮训练
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            valid_windows = 0
            
            # 遍历每个小时窗口
            for window_idx, (X, y) in enumerate(tqdm(window_data, desc="处理窗口")):
                if len(X) < min_samples_per_window:
                    print(f"  窗口 {window_idx+1} 样本数 {len(X)} 不足 {min_samples_per_window}，跳过")
                    continue
                
                # 划分训练集和验证集
                train_size = max(int((1 - validation_ratio) * len(X)), 1)  # 确保至少有1个训练样本
                val_size = len(X) - train_size
                
                if train_size <= 0:
                    print(f"  窗口 {window_idx+1} 训练集为空，跳过")
                    continue
                    
                if val_size <= 0:
                    # 如果验证集为空，则使用一小部分训练集作为验证集
                    val_size = max(1, int(0.1 * len(X)))  # 至少一个样本，最多10%
                    train_size = len(X) - val_size
                
                print(f"  窗口 {window_idx+1} 训练集大小: {train_size}, 验证集大小: {val_size}")
                
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
                
                # 创建数据加载器
                train_dataset = DrivingDataset(X_train, y_train)
                val_dataset = DrivingDataset(X_val, y_val)
                
                # 检查数据集是否为空
                if len(train_dataset) == 0:
                    print(f"  窗口 {window_idx+1} 训练数据集为空，跳过")
                    continue
                
                if len(val_dataset) == 0:
                    print(f"  窗口 {window_idx+1} 验证数据集为空，跳过")
                    continue
                
                train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)))
                
                # 训练当前窗口
                window_train_loss = 0.0
                window_val_loss = 0.0
                
                for _ in range(window_epochs):
                    # 训练阶段
                    self.model.train()
                    batch_train_loss = 0.0
                    batch_count = 0
                    
                    for inputs, targets in train_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        # 前向传播 - 为每个批次重新初始化隐藏状态
                        self.optimizer.zero_grad()
                        # 不传递hidden状态，让模型自己初始化
                        outputs, _ = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        # 反向传播和优化
                        loss.backward()
                        self.optimizer.step()
                        
                        batch_train_loss += loss.item()
                        batch_count += 1
                    
                    if batch_count > 0:
                        window_train_loss += batch_train_loss / batch_count
                
                if window_epochs > 0:
                    window_train_loss /= window_epochs
                
                # 验证阶段
                self.model.eval()
                batch_val_loss = 0.0
                batch_count = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        # 不传递hidden状态，让模型自己初始化
                        outputs, _ = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        batch_val_loss += loss.item()
                        batch_count += 1
                
                if batch_count > 0:
                    window_val_loss = batch_val_loss / batch_count
                    
                epoch_train_loss += window_train_loss
                epoch_val_loss += window_val_loss
                valid_windows += 1
                
                # 输出每个窗口的训练信息
                print(f"  窗口 {window_idx+1}/{len(window_data)}, "
                      f"训练损失: {window_train_loss:.4f}, 验证损失: {window_val_loss:.4f}")
            
            # 检查是否有有效的窗口
            if valid_windows == 0:
                print("没有有效的窗口可以训练，跳过此轮次")
                continue
                
            # 计算整个epoch的平均损失
            epoch_train_loss /= valid_windows
            epoch_val_loss /= valid_windows
            
            train_history['train_losses'].append(epoch_train_loss)
            train_history['val_losses'].append(epoch_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"平均训练损失: {epoch_train_loss:.4f}, 平均验证损失: {epoch_val_loss:.4f}")
            
            # 早停检查
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
                print("  保存当前最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if os.path.exists('best_gru_model.pth'):
            self.model.load_state_dict(torch.load('best_gru_model.pth'))
            print("已加载最佳模型")
        else:
            print("没有找到最佳模型文件，使用当前模型")
        
        return train_history
    
    def evaluate_sliding_window(self, csv_file_path, start_hour=None, num_hours=1, window_step_size=None):
        """
        使用滑动窗口评估模型
        
        参数:
        - csv_file_path: CSV文件或目录路径
        - start_hour: 开始评估的小时索引 (可选，默认使用最后一个小时)
        - num_hours: 要评估的小时数
        - window_step_size: 窗口滑动步长，默认为半个窗口大小（50%重叠）
        
        返回:
        - 测试损失和预测结果
        """
        # 准备数据
        scaled_df = self.prepare_data_sliding_window(csv_file_path)
        
        # 如果未指定步长，使用默认值
        if window_step_size is None:
            window_step_size = self.samples_per_hour // 2  # 默认使用半个窗口大小作为步长
        
        # 计算总窗口数
        total_samples = len(scaled_df)
        max_windows = (total_samples - self.samples_per_hour) // window_step_size + 1
        
        # 如果没有指定开始小时，则使用最后num_hours个小时
        if start_hour is None:
            start_hour = max(0, max_windows - num_hours)
        
        # 确保不超出范围
        end_hour = min(start_hour + num_hours, max_windows)
        hour_windows = list(range(start_hour, end_hour))
        
        # 创建测试窗口数据
        test_windows = self.create_hour_windows(scaled_df, hour_windows, min_sequence_length=1, step_size=window_step_size)
        
        if not test_windows:
            print("没有足够的数据来创建测试窗口！")
            return None, None, None
        
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_actuals = []
        
        for window_idx, (X, y) in enumerate(test_windows):
            # 创建数据加载器
            test_dataset = DrivingDataset(X, y)
            test_loader = DataLoader(test_dataset, batch_size=min(self.batch_size, len(test_dataset)))
            
            window_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 不传递hidden状态，让模型自己初始化
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    window_loss += loss.item()
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(targets.cpu().numpy())
            
            window_loss /= len(test_loader)
            test_loss += window_loss
            
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)
            
            print(f"窗口 {window_idx+1}/{len(test_windows)}, 测试损失: {window_loss:.4f}")
        
        test_loss /= len(test_windows)
        print(f"平均测试损失: {test_loss:.4f}")
        
        # 计算均方根误差（RMSE）
        rmse = np.sqrt(((np.array(all_predictions) - np.array(all_actuals)) ** 2).mean())
        print(f"Root Mean Square Error: {rmse:.4f}")
        
        return test_loss, all_predictions, all_actuals
    
    def prepare_data(self, csv_file, sample_interval=1):
        """
        从CSV文件准备数据（保留原有方法）
        
        参数:
        - csv_file: CSV文件路径
        - sample_interval: 采样间隔（每隔多少秒采样一次）
        
        返回:
        - X_train, X_test, y_train, y_test: 训练集和测试集的特征和标签
        """
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 选择需要的列
        selected_features = ['Blink Count', 'Blink Rate(per minute)', 'Yawn Count', 
                             'Yawn Rate(per hour)', 'Fatigue Score']
        df = df[selected_features]
        
        # 处理缺失值
        df = df.ffill()
        
        # 创建序列和标签
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length - self.prediction_horizon):
            # 当前序列
            seq = df.iloc[i:i+self.sequence_length].values
            # 预测目标 (20分钟后的疲劳分数)
            target = df.iloc[i+self.sequence_length+self.prediction_horizon-1]['Fatigue Score']
            
            X.append(seq)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # 数据标准化
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val, y_val, num_epochs=100, patience=10):
        """
        训练模型（保留原有方法）
        
        参数:
        - X_train, y_train: 训练数据
        - X_val, y_val: 验证数据
        - num_epochs: 训练轮次
        - patience: 早停的耐心值
        
        返回:
        - 训练历史记录
        """
        train_dataset = DrivingDataset(X_train, y_train)
        val_dataset = DrivingDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 用于早停的变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_gru_model.pth'))
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def evaluate(self, X_test, y_test):
        """
        评估模型（保留原有方法）
        
        参数:
        - X_test, y_test: 测试数据
        
        返回:
        - 测试损失和预测结果
        """
        self.model.eval()
        test_dataset = DrivingDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        
        # 计算均方根误差（RMSE）
        rmse = np.sqrt(((np.array(predictions) - np.array(actuals)) ** 2).mean())
        print(f'Root Mean Square Error: {rmse:.4f}')
        
        return test_loss, predictions, actuals
    
    def predict(self, sequence):
        """
        使用训练好的模型进行预测（保留原有方法）
        
        参数:
        - sequence: 输入序列，形状为 (sequence_length, input_size)
        
        返回:
        - 预测的疲劳分数
        """
        self.model.eval()
        
        # 标准化输入序列
        sequence_scaled = self.scaler.transform(sequence)
        
        # 转换为PyTorch张量并增加批次维度
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 不传递hidden状态，让模型自己初始化
            prediction, _ = self.model(sequence_tensor)
            
        return prediction.item()
    
    def save_model(self, path='gru_fatigue_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'samples_per_hour': self.samples_per_hour
        }, path)
        print(f'模型已保存到 {path}')
    
    def load_model(self, path='gru_fatigue_model.pth'):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            
            # 重建模型
            self.input_size = checkpoint['input_size']
            self.hidden_size = checkpoint['hidden_size']
            self.num_layers = checkpoint['num_layers']
            self.sequence_length = checkpoint['sequence_length']
            self.prediction_horizon = checkpoint['prediction_horizon']
            self.samples_per_hour = checkpoint.get('samples_per_hour', 60)  # 兼容旧版本
            
            self.model = GRUModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1
            ).to(self.device)
            
            # 加载模型参数
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler = checkpoint['scaler']
            
            print(f'模型已从 {path} 加载')
        else:
            print(f'模型文件 {path} 不存在')
    
    def visualize_results(self, history, predictions=None, actuals=None):
        """
        可视化训练结果和预测
        
        参数:
        - history: 训练历史记录
        - predictions: 预测值
        - actuals: 实际值
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 如果提供了预测和实际值，则绘制对比图
        if predictions is not None and actuals is not None:
            plt.subplot(1, 2, 2)
            plt.scatter(actuals, predictions, alpha=0.5)
            
            # 添加理想预测线（y=x）
            min_val = min(min(predictions), min(actuals))
            max_val = max(max(predictions), max(actuals))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Fatigue Score')
            plt.ylabel('Predicted Fatigue Score')
            plt.title('Prediction vs Actual')
        
        plt.tight_layout()
        plt.savefig('fatigue_prediction_results.png')
        plt.show()

def main_sliding_window(csv_file_path):
    """
    使用滑动窗口方法训练和评估模型
    
    参数:
    - csv_file_path: CSV文件路径
    """
    # 初始化预测系统
    fatigue_system = FatiguePredictionSystem(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        sequence_length=1800,  # 使用30分钟（1800秒）的数据作为输入序列，不是1小时
        prediction_horizon=1200,  # 预测20分钟（1200秒）后的疲劳分数
        samples_per_hour=3600  # 每小时3600个样本点（每秒1个）
    )
    
    try:
        # 使用滑动窗口训练模型
        print("使用滑动窗口训练模型...")
        history = fatigue_system.train_with_sliding_window(
            csv_file_path, 
            num_epochs=10,  # 整个数据集的训练轮次
            window_epochs=2,  # 每个窗口的训练轮次
            validation_ratio=0.2, 
            patience=5,
            min_samples_per_window=1,  # 降低每个窗口的最小样本数要求
            window_step_size=900  # 使用15分钟作为滑动步长，创建重叠窗口
        )
        
        if history:
            # 评估模型
            print("\n评估模型...")
            test_loss, predictions, actuals = fatigue_system.evaluate_sliding_window(
                csv_file_path,
                start_hour=None,  # 使用最后一个小时
                num_hours=1,
                window_step_size=900  # 使用与训练相同的步长
            )
            
            # 可视化结果
            if predictions and actuals:
                fatigue_system.visualize_results(history, predictions, actuals)
            
            # 保存模型
            fatigue_system.save_model('sliding_window_gru_model.pth')
            print("训练和评估完成！")
        else:
            print("训练未产生有效的历史记录，请检查数据")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main(csv_file_path):
    """
    主函数，用于训练和评估模型（保留原有方法）
    
    参数:
    - csv_file_path: CSV文件路径
    """
    # 初始化预测系统
    fatigue_system = FatiguePredictionSystem(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        sequence_length=1800,  # 使用30分钟（1800秒）的数据作为输入序列
        prediction_horizon=1200,  # 预测20分钟（1200秒）后的疲劳分数
        samples_per_hour=3600  # 每小时3600个样本点（每秒1个）
    )
    
    # 准备数据
    X_train, X_test, y_train, y_test = fatigue_system.prepare_data(csv_file_path)
    
    # 训练模型
    history = fatigue_system.train(X_train, y_train, X_test, y_test, num_epochs=100, patience=10)
    
    # 评估模型
    _, predictions, actuals = fatigue_system.evaluate(X_test, y_test)
    
    # 可视化结果
    fatigue_system.visualize_results(history, predictions, actuals)
    
    # 保存模型
    fatigue_system.save_model()

if __name__ == "__main__":
    # 使用示例
    csv_file_path = r"D:\Project\guaduation_project\data\fatigue_dataset"  # 替换为实际CSV文件路径
    
    # 使用滑动窗口方法训练和评估
    main_sliding_window(csv_file_path)
    
    # 如果需要使用原始方法，可以取消下面的注释
    # main(csv_file_path)
