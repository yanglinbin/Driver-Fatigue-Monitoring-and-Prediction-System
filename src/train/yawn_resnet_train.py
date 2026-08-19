import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models, datasets


# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 数据加载函数 - 使用ImageFolder处理两个类别的文件夹结构
def load_data(data_dir, batch_size=32, train_ratio=0.8):
    print(f"Loading data from {data_dir}")
    
    # 数据预处理和增强 - 移除随机裁剪以减少过拟合
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载整个数据集
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # 计算训练集和验证集大小
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 随机分割数据集
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 应用不同的变换
    train_dataset = DatasetTransformer(train_dataset, train_transform)
    val_dataset = DatasetTransformer(val_dataset, val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    
    return train_loader, val_loader, class_names

# 辅助类用于应用变换到已分割的数据集
class DatasetTransformer(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 构建ResNet18模型
def build_model(num_classes=2, pretrained=True, dropout_rate=0.5):
    # 在新版PyTorch中，pretrained参数已更改为weights
    try:
        # 新版本PyTorch (>= 0.13)
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    except TypeError:
        # 旧版本PyTorch
        model = models.resnet18(pretrained=pretrained)
    
    # 修改最后一层以适应分类任务，并添加Dropout以减少过拟合
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# 训练一个epoch - 使用软标签
def train_epoch(model, dataloader, criterion, optimizer, device, label_smoothing=0.1):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 创建软标签
        batch_size = labels.size(0)
        one_hot_labels = torch.zeros(batch_size, 2).to(device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        soft_labels = one_hot_labels * (1 - label_smoothing) + label_smoothing / 2
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, soft_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # 使用阈值0.3来预测，更倾向于no_yawn
        probabilities = torch.softmax(outputs, dim=1)
        preds = (probabilities[:, 1] > 0.3).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='binary')
    epoch_recall = recall_score(all_labels, all_preds, average='binary')
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

# 验证一个epoch - 使用相同的阈值
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 创建软标签用于计算损失
            batch_size = labels.size(0)
            one_hot_labels = torch.zeros(batch_size, 2).to(device)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, one_hot_labels)  # 验证时使用硬标签计算损失
            
            running_loss += loss.item() * inputs.size(0)
            
            # 使用阈值0.3来预测，更倾向于no_yawn
            probabilities = torch.softmax(outputs, dim=1)
            preds = (probabilities[:, 1] > 0.3).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='binary')
    epoch_recall = recall_score(all_labels, all_preds, average='binary')
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

# 绘制训练曲线
def plot_metrics(train_metrics, val_metrics, metric_name, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{metric_name.lower()}_curve.png'))
    plt.close()

# 主训练函数
def train_model(data_dir, save_dir, num_epochs=30, batch_size=32, learning_rate=0.0001,
                weight_decay=1e-4, pretrained=True, dropout_rate=0.5, label_smoothing=0.1):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置随机种子
    set_seed()
    
    # 加载数据
    train_loader, val_loader, class_names = load_data(data_dir, batch_size)
    num_classes = len(class_names)
    print(f"Training for {num_classes} classes: {class_names}")
    
    # 构建模型
    model = build_model(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 改进的学习率调度器 - 使用CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 训练和验证指标记录
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    # 记录最佳模型
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # 训练循环
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, label_smoothing
        )
        
        # 验证阶段
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step()
        
        # 记录指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with F1 score: {best_val_f1:.4f}")
        
        # 打印训练信息
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {time_elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        print("-" * 60)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 绘制训练曲线
    plot_metrics(history['train_loss'], history['val_loss'], 'Loss', save_dir)
    plot_metrics(history['train_acc'], history['val_acc'], 'Accuracy', save_dir)
    plot_metrics(history['train_f1'], history['val_f1'], 'F1 Score', save_dir)
    
    print(f"Training completed. Best validation F1 score: {best_val_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    
    return model, history

# 预测函数
def predict(model, image, transform, device, threshold=0.3):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        # 使用阈值0.3来预测，更倾向于no_yawn
        pred = 1 if probabilities[0, 1] > threshold else 0
    
    return pred

# 主函数
if __name__ == "__main__":
    # 配置参数
    data_dir = "D:\\Project\\guaduation_project\\data\\datasets\\yawn"  # 数据目录
    save_dir = "models/yawn_resnet/new"  # 模型保存目录
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.000001
    weight_decay = 1e-4
    pretrained = True  # 是否使用预训练模型
    dropout_rate = 0.7  # Dropout率
    label_smoothing = 0.1  # 标签平滑
    
    # 训练模型
    model, history = train_model(
        data_dir=data_dir,
        save_dir=save_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing
    ) 