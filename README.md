# 驾驶员疲劳监测与预测系统

基于深度学习的驾驶员疲劳驾驶检测与预测系统（毕业设计项目）。系统通过摄像头实时采集驾驶员面部视频流，利用 MediaPipe 提取面部关键点，结合 ResNet-18 分类模型检测眼睛开闭与哈欠状态，并通过 SVD 平面拟合算法估计头部姿态，最终使用加权评分对疲劳等级进行评估，并基于 GRU 时序模型预测未来的疲劳趋势。

## 功能特性

- **眼睛状态检测**：通过 MediaPipe FaceMesh 关键点裁剪左右眼区域，输入 ResNet-18 二分类模型判断睁眼/闭眼
- **眨眼与持续闭眼检测**：基于闭眼→睁眼状态转换识别眨眼，持续闭眼超过阈值（默认 2 秒）触发报警
- **哈欠检测**：裁剪嘴部区域，输入 ResNet-18 二分类模型判断哈欠状态，持续打哈欠超过阈值（默认 2 秒）才计为一次哈欠
- **头部姿态估计**：使用 7 个面部关键点进行 SVD 平面拟合，计算欧拉角（Roll/Pitch/Yaw），并识别点头、低头行为
- **疲劳等级评估**：融合眨眼频率（次/分钟）与哈欠频率（次/小时），通过加权评分输出 5 级疲劳状态
- **数据记录与可视化**：多线程架构实时记录每秒驾驶状态到 CSV，并同步展示 Tkinter + Matplotlib 实时曲线
- **疲劳趋势预测**：基于记录的时序数据训练 GRU 模型，预测未来（默认 20 分钟）的疲劳分数
- **模型评估工具**：支持在图像数据集、文件夹和视频数据集上评估眼睛/哈欠模型的准确率、精确率、召回率、F1 与混淆矩阵

## 系统架构

```
摄像头视频流
    │
    ▼
MediaPipe FaceMesh（468 个面部关键点）
    ├── 裁剪左右眼区域 ──► ResNet-18（睁眼/闭眼）──► 眨眼 / 持续闭眼计数
    ├── 裁剪嘴部区域 ──► ResNet-18（哈欠/非哈欠）──► 哈欠计数
    └── SVD 平面拟合 ──► 欧拉角 (Roll/Pitch/Yaw) ──► 点头 / 低头检测
    │
    ▼
FatigueAnalyzer（滑动窗口 + 加权评分）
    │
    ├──► 疲劳等级（正常 / 轻度 / 中度 / 重度 / 危险）
    ├──► 每秒写入 CSV（FatigueDataRecorder）
    └──► GRU 时序模型 ──► 预测未来疲劳分数
```

### 疲劳等级定义

| 等级 | 说明 |
| --- | --- |
| NORMAL (0) | 正常 |
| MILD (1) | 轻度疲劳 |
| MODERATE (2) | 中度疲劳 |
| SEVERE (3) | 重度疲劳 |
| DANGEROUS (4) | 危险驾驶 |

评分逻辑（详见 `src/fatigue_judgment/fatigue_analyzer.py`）：

- 眨眼频率权重 0.6，哈欠频率权重 0.4，线性映射到 0–100 分
- 持续闭眼或低头状态各额外加 15 分
- 同时闭眼与低头 → 直接判定为危险驾驶（100 分）；持续闭眼且近期有点头 → 直接判定为重度疲劳（90 分）
- 综合评分 ≥85 重度、≥65 中度、≥40 轻度，否则正常

## 项目结构

```
.
├── data/
│   ├── datasets/              # 眼睛/嘴部图像数据集
│   │   ├── eyes/              # closed_eyes(1962 张) / open_eyes(726 张)
│   │   └── yawn/              # no_yawn(3315 张) / yawn(3419 张)
│   ├── fatigue_data/          # 系统实测记录的疲劳数据（每秒一行 CSV）
│   └── fatigue_dataset/       # 虚拟生成的疲劳时序数据（50 份，每份 4–6 小时）
├── models/                    # 训练好的模型权重与网络架构图
├── src/
│   ├── train/                 # ResNet-18 训练脚本（眼睛 / 哈欠）
│   ├── validation/            # 实时检测验证器与模型评估器
│   ├── fatigue_judgment/      # 疲劳分析器与数据记录器
│   └── prediction_system/     # GRU 疲劳预测系统
├── utils/                     # 眼睛/嘴部区域裁剪、虚拟数据生成
├── main.py                    # 空文件（入口待完善）
├── system_description.md      # 系统设计与实现说明
└── pyproject.toml             # 项目元数据（依赖列表不完整，见“已知问题”）
```

> 注：`data/`、`test/`（测试图片）等大数据目录通过 `.gitignore` 排除，不随代码上传；`models/` 中的模型权重会上传。

## 环境要求与安装

### 环境要求

- Python 3.8+（仓库内 `.python-version` 为 3.11.9；MediaPipe 0.10 不支持 Python 3.12+）
- 摄像头（实时检测时使用；也可以改为视频文件）
- 可选：NVIDIA GPU + CUDA（头部姿态模块使用 `cupy`，训练与 GRU 推理可自动使用 CUDA）

### 安装依赖

代码实际使用的依赖包括：

```bash
pip install opencv-python mediapipe numpy matplotlib \
            torch torchvision scikit-learn pandas tqdm \
            cupy-cuda12x pillow
```

> `cupy-cuda12x` 需根据本机 CUDA 版本选择对应的包名；纯 CPU 环境可跳过 cupy，但需要修改 `src/validation/pose_detection_validation.py` 与 `src/validation/integrated_detection_validator.py` 中的 cupy 调用。

## 快速开始

### 1. 数据集准备

眼睛/嘴部区域裁剪工具（需要原始人脸图像）：

```bash
python utils/eyes_tiqu.py        # 提取左右眼区域
python utils/mouse_tiqu.py       # 提取嘴部（自定义）区域
```

虚拟疲劳时序数据生成（供 GRU 训练使用，默认生成 1000 份，脚本末尾的批量输出路径为硬编码）：

```bash
python "utils/Virtual data generation.py"
```

### 2. 模型训练

眼睛状态二分类（ResNet-18 + 标签平滑 + OneCycleLR）：

```bash
python src/train/eyes_resnet_train.py
```

哈欠状态二分类（ResNet-18 + 软标签 + CosineAnnealingLR）：

```bash
python src/train/yawn_resnet_train.py
```

GRU 疲劳预测模型（滑动窗口训练 + 早停 + 评估）：

```bash
python src/prediction_system/gru_model.py
```

### 3. 实时检测

```bash
# 眼睛状态检测
python src/validation/eye_detection_validator.py

# 哈欠检测
python src/validation/yawn_detection_validator.py

# 头部姿态检测
python src/validation/pose_detection_validation.py

# 集成检测（眼睛 + 哈欠 + 头部姿态 + 疲劳评分 + 数据记录，按 R 键开始记录、ESC 退出）
python src/validation/integrated_detection_validator.py
```

### 4. 模型评估

```bash
python src/validation/evaluator/eye_detection_evaluator.py \
    --model_path models/eyes_resnet_18_l.pth --data_dir <数据集目录>

python src/validation/evaluator/yawn_detection_evaluator.py \
    --model_path models/yawn_resnet_18.pth --data_dir <数据集目录>
```

支持 `--video_dir`（视频数据集）与 `--image_folders_dir`（文件夹直接评估）参数，详见脚本内 argparse 定义。

## 关键实现说明

### 面部关键点索引

- 左眼：`[35, 70, 52, 55, 188, 120, 117]`
- 右眼：`[285, 295, 300, 265, 346, 349, 412]`
- 嘴部：`[57, 165, 164, 391, 287, 406, 18, 182, 57]`
- 姿态平面拟合：`[117, 346, 151, 9, 4, 23, 253]`

### 模型结构

- **眼睛模型**：ResNet-18 预训练权重 + 池化前 Dropout(0.4) + 双层 MLP 分类头（512 隐藏层，双重 Dropout(0.5)），输出 2 类
- **哈欠模型**：ResNet-18 预训练权重 + Dropout 分类头（默认 0.7），输出 2 类，训练/推理使用 0.3 概率阈值更倾向“非哈欠”
- **GRU 预测模型**：输入 5 个特征（Blink Count、Blink Rate、Yawn Count、Yawn Rate、Fatigue Score），2 层 GRU（hidden=64，层间 Dropout 0.2），输出 1 维疲劳分数；默认输入序列 30 分钟、预测未来 20 分钟，MSE 损失 + Adam 优化器

### 行为识别规则（`fatigue_data_recorder.py`）

- 眨眼：历史窗口中“闭眼→睁眼”转换，冷却时间 0.6 秒
- 持续闭眼：双眼闭合持续 ≥2 秒
- 哈欠：哈欠状态持续 ≥2 秒计一次
- 低头：Pitch 超过历史均值 +15° 且持续 ≥2 秒
- 点头：Pitch 短暂超过阈值后回落（未达到低头时长）

## 数据集说明

| 数据集 | 内容 | 规模 |
| --- | --- | --- |
| `data/datasets/eyes` | 眼睛区域图像 | closed_eyes 1962 张、open_eyes 726 张 |
| `data/datasets/yawn` | 嘴部区域图像 | no_yawn 3315 张、yawn 3419 张 |
| `data/fatigue_data` | 系统实测记录（每秒一行） | 多次运行的 CSV |
| `data/fatigue_dataset` | 虚拟生成的驾驶疲劳时序数据 | 50 份 CSV，每份约 4–6 小时、1 秒/条 |

疲劳数据 CSV 列：`Timestamp, Blink Count, Blink Rate(per minute), Yawn Count, Yawn Rate(per hour), Nod Status, Head Down Status, Fatigue Score, Fatigue Level`。

## 模型文件（`models/`）

- `eyes_resnet_18.pth`：眼睛状态 ResNet-18 权重
- `yawn_best_model.pth`：哈欠状态 ResNet-18 最佳权重
- `fatigue_trend_model/best_fatigue_trend_predictor.pth`：疲劳趋势预测（GRU）权重
- `eyes_resnet.py` / `yawn_resnet18.py` / `gru.py` / `resnet18.py`：网络结构示意图绘制脚本
- 其余为网络架构图（PDF/PNG）

## 已知问题与注意事项

1. **硬编码绝对路径**：多数脚本中存在 `D:\Project\guaduation_project\...`（或 `D:/Project/...`）等绝对路径，例如模型路径、CSV 输出目录（`D:/Project/guaduation_project/data/fatigue_data`）、测试视频路径等，运行前需要改为本机实际路径。
2. **依赖声明不完整**：`pyproject.toml` 仅声明了 opencv-python、mediapipe、numpy、matplotlib，但代码实际还依赖 PyTorch、torchvision、scikit-learn、pandas、tqdm、cupy、Pillow。
3. **cupy 依赖**：头部姿态与集成检测模块直接 import cupy，无 CUDA 环境的机器需要替换为 NumPy 实现。
4. **入口文件为空**：`main.py` 目前是空文件，实时检测入口请使用 `src/validation/integrated_detection_validator.py`。

## 相关文档

- [system_description.md](system_description.md)：系统架构、模块与算法说明
- `models/*.png|pdf`：ResNet-18 / GRU 网络结构图
