import os
import time
from collections import deque

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


# 标签平滑损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# 数据加载函数
def load_data(data_dir, batch_size=32, train_ratio=0.8):
    print(f"Loading data from {data_dir}")

    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(3, sigma=(0.1, 0.5))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.RandomErasing(p=0.2)
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


# 构建模型
def build_model(num_classes=2, pretrained=True):
    try:
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=pretrained)

    # 添加dropout层
    model.avgpool = nn.Sequential(
        nn.Dropout(0.4),
        model.avgpool
    )

    # 修改全连接层，添加dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model


# 训练一个epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='binary')
    epoch_recall = recall_score(all_labels, all_preds, average='binary')
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1


# 验证一个epoch
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
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
def train_model(data_dir, save_dir, num_epochs=50, batch_size=64, learning_rate=0.001,
                weight_decay=5e-2, pretrained=True):
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
    model = build_model(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # 使用标签平滑的损失函数
    criterion = LabelSmoothingLoss(smoothing=0.05)

    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

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
            model, train_loader, criterion, optimizer, device
        )

        # 验证阶段
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )

        # 更新学习率
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
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {time_elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
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
def predict(model, image, transform, device):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return preds.item()


# 主函数
if __name__ == "__main__":
    # 配置参数
    data_dir = "D:\\Project\\guaduation_project\\data\\datasets\\eyes"
    save_dir = "models/eyes_resnet"
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.00001
    weight_decay = 5e-2
    pretrained = True

    # 训练模型
    model, history = train_model(
        data_dir=data_dir,
        save_dir=save_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pretrained=pretrained
    )

# 眼睛状态判断
eye_state_history = deque(maxlen=10)
recent_states = list(eye_state_history)[-5:]
if len(recent_states) > 0:
    recent_weight = 0.7
    history_weight = 0.3
    recent_score = sum(recent_states) / len(recent_states)
    history_score = sum(eye_state_history) / len(eye_state_history)
    final_score = recent_weight * recent_score + history_weight * history_score
    is_closed = final_score > 0.3  # 使用0.3作为阈值