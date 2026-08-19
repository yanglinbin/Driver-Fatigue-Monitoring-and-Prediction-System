# 驾驶员疲劳监测与预测系统实现

## 一、系统架构

系统主要实现了基于实时视频流的驾驶员疲劳检测与预测。主要检测眼睛眨眼频率、打哈欠频率和头部姿态信息，使用多模态信息融合评估驾驶员疲劳状态。系统包含以下主要组件：

1. **模型训练模块**：用于训练各种特征检测的深度学习模型
2. **特征验证模块**：用于实时验证和检测驾驶员的面部特征
3. **疲劳判断模块**：基于检测到的特征进行疲劳分析和评估
4. **疲劳预测系统**：基于历史数据预测未来疲劳趋势

![系统架构图](系统架构.png)

系统使用摄像头捕捉实时视频流，通过Mediapipe关键点检测技术对眼部和嘴部进行裁剪，将裁剪后的图像数据输入到训练好的ResNet-18模型中进行识别，判断眼睛和嘴巴的实时状态。通过连续帧判断驾驶员是否眨眼和打哈欠，同时数据记录模块会记录每一秒的驾驶员状态。系统通过统计眨眼和打哈欠次数，使用滑动窗口来计算每分钟眨眼频率和每小时打哈欠频率，使用加权评分系统对驾驶员的疲劳程度进行打分。

## 二、模型训练模块

该模块负责训练用于检测眼睛状态和哈欠状态的ResNet模型。

### 眼睛状态检测模型

```python
def build_model(num_classes=2, pretrained=True):
    # 使用预训练的ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 添加dropout层防止过拟合
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
```

### 哈欠状态检测模型

```python
def build_model(num_classes=2, pretrained=True):
    # 使用预训练的ResNet18
    try:
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=pretrained)
    
    # 添加dropout层防止过拟合
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
```

### 数据增强策略

```python
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
])
```

### 关键特性

1. 基于ResNet18架构
2. 使用ImageNet预训练权重
3. 添加Dropout层防止过拟合
4. 使用标签平滑技术提高模型泛化能力
5. 丰富的数据增强技术提高模型鲁棒性

## 三、特征验证模块

### 1. 眼睛状态检测验证器

用于从视频流中检测驾驶员的眼睛状态（睁眼或闭眼）。系统使用MediaPipe提取眼部关键点，再进行区域裁剪，将裁剪后的图像输入到ResNet-18模型中判断眼睛状态。

```python
def extract_eye_regions(image, face_landmarks, left_eye_indices, right_eye_indices, padding=5):
    """从图像中提取左右眼区域"""
    h, w, _ = image.shape
    
    # 获取左眼和右眼的关键点坐标
    left_eye_points = [(int(face_landmarks.landmark[idx].x * w), 
                        int(face_landmarks.landmark[idx].y * h)) 
                        for idx in left_eye_indices]
    
    right_eye_points = [(int(face_landmarks.landmark[idx].x * w), 
                        int(face_landmarks.landmark[idx].y * h)) 
                        for idx in right_eye_indices]
    
    # 计算边界框坐标
    def get_eye_bbox(points, padding):
        min_x = max(0, min(p[0] for p in points) - padding)
        min_y = max(0, min(p[1] for p in points) - padding)
        max_x = min(w, max(p[0] for p in points) + padding)
        max_y = min(h, max(p[1] for p in points) + padding)
        return min_x, min_y, max_x, max_y
    
    # 计算左右眼区域的边界框
    left_eye_min_x, left_eye_min_y, left_eye_max_x, left_eye_max_y = get_eye_bbox(left_eye_points, padding)
    right_eye_min_x, right_eye_min_y, right_eye_max_x, right_eye_max_y = get_eye_bbox(right_eye_points, padding)
    
    # 提取左右眼区域
    left_eye_region = image[left_eye_min_y:left_eye_max_y, left_eye_min_x:left_eye_max_x]
    right_eye_region = image[right_eye_min_y:right_eye_max_y, right_eye_min_x:right_eye_max_x]
    
    # 检查区域是否有效
    if left_eye_region.size == 0 or right_eye_region.size == 0:
        return None, None
    
    return left_eye_region, right_eye_region
```

### 2. 哈欠检测验证器

检测驾驶员的哈欠状态，通过提取嘴部区域并使用特定模型进行分类。系统同样使用MediaPipe提取嘴部关键点，再进行区域裁剪，将裁剪后的图像输入到ResNet-18模型中判断嘴部状态。

```python
def extract_mouth_region(image, face_landmarks, mouth_indices, padding=10):
    """从图像中提取嘴部区域"""
    h, w, _ = image.shape
    
    # 获取嘴部的关键点坐标
    mouth_points = [(int(face_landmarks.landmark[idx].x * w), 
                      int(face_landmarks.landmark[idx].y * h)) 
                     for idx in mouth_indices]
    
    # 计算嘴部区域的边界框
    mouth_min_x = max(0, min(p[0] for p in mouth_points) - padding)
    mouth_min_y = max(0, min(p[1] for p in mouth_points) - padding)
    mouth_max_x = min(w, max(p[0] for p in mouth_points) + padding)
    mouth_max_y = min(h, max(p[1] for p in mouth_points) + padding)
    
    # 提取嘴部区域
    mouth_region = image[mouth_min_y:mouth_max_y, mouth_min_x:mouth_max_x]
    
    # 检查区域是否有效
    if mouth_region.size == 0:
        return None
    
    return mouth_region

def predict_yawn_state(model, mouth_region, transform, device, class_names, threshold=0.85):
    """预测是否打哈欠，使用阈值更倾向于no_yawn"""
    if mouth_region is None or mouth_region.size == 0:
        return "Unknown", 0.0
    
    # 转换为PIL图像和预测
    try:
        pil_image = Image.fromarray(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB))
        # 应用变换并转换为模型输入
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            yawn_prob = probabilities[0, 1].item()
            
            # 使用阈值进行判断
            prediction = 1 if yawn_prob > threshold else 0
            confidence = yawn_prob if prediction == 1 else 1.0 - yawn_prob
        
        return class_names[prediction], confidence
    except Exception as e:
        print(f"Error in yawn state prediction: {e}")
        return "Unknown", 0.0
```

### 3. 头部姿态检测验证器

头部姿态检测模块会检测头部姿态，输出欧拉角信息（roll、pitch、yaw）来分析头部倾斜和点头动作。在系统中，重点关注pitch的数据变化，在系统启动以后，会持续记录pitch值，并且计算平均值，该平均值即为驾驶员正常驾驶时的头部俯仰角数据。如果驾驶员处于疲劳状态，发生点头或者持续低头行为，都会被系统记录，评估是否处于危险驾驶行为。

```python
def fit_plane(points):
    """
    使用 SVD 拟合一个平面，返回平面质心和法向量
    points: cupy 数组，形状 (N,3)
    """
    points = cp.asarray(points)  # 确保输入是cupy数组
    centroid = cp.mean(points, axis=0)
    A = points - centroid
    # 使用SVD计算法向量
    _, _, Vt = svd(A)
    normal = Vt[-1]
    # 保证 z 分量为正
    if normal[2] < 0:
        normal = -normal
    return cp.asnumpy(centroid), cp.asnumpy(normal)  # 转换回numpy数组用于显示

def get_landmark_coords(landmarks, idx, image_shape):
    """
    将 mediapipe 的归一化坐标转换为图像坐标
    """
    h, w = image_shape[:2]
    lm = landmarks[idx]
    return cp.array([lm.x * w, lm.y * h, lm.z * w])  # z 按宽度估算

def compute_euler_angles(x_axis, y_axis, z_axis):
    """
    使用旋转矩阵计算欧拉角（Roll, Pitch, Yaw）
    """
    # 确保所有输入都是cupy数组
    x_axis = cp.asarray(x_axis)
    y_axis = cp.asarray(y_axis)
    z_axis = cp.asarray(z_axis)
    
    # 构造旋转矩阵
    R = cp.column_stack((x_axis, y_axis, z_axis))
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]
    
    # 计算欧拉角
    original_roll = math.degrees(cp.arctan2(r32, r33))
    original_pitch = math.degrees(cp.arcsin(-r31))
    original_yaw = math.degrees(cp.arctan2(r21, r11))
    
    # 映射关系调整
    roll = original_yaw    # 原来的yaw现在是roll
    pitch = original_roll  # 原来的roll现在是pitch
    yaw = original_pitch   # 原来的pitch现在是yaw
    
    return float(roll), float(pitch), float(yaw)  # 转换为Python float
```

在集成检测系统中的头部姿态检测实现：

```python
# 姿态估计
# 收集平面拟合点
pts_plane = cp.array([
    get_landmark_coords(face_landmarks.landmark, idx, frame.shape)
    for idx in PLANE_IDX
])

# 拟合平面，得到平面质心和法向量
centroid, normal = fit_plane(pts_plane)

# 计算局部坐标系
# X轴方向：使用两点方向定义
vec = pts_plane[1] - pts_plane[0]
x_axis = vec / max(float(cp.linalg.norm(vec)), 1e-5)  # 避免除零

# Z轴方向：法向量
z_axis = cp.asarray(normal)

# Y轴方向：Z叉乘X
y_axis = cp.cross(z_axis, x_axis)
y_axis = y_axis / max(float(cp.linalg.norm(y_axis)), 1e-5)  # 避免除零

# 重新计算X轴以确保正交性
x_axis = cp.cross(y_axis, z_axis)
x_axis = x_axis / max(float(cp.linalg.norm(x_axis)), 1e-5)

# 计算欧拉角
roll, pitch, yaw = compute_euler_angles(x_axis, y_axis, z_axis)
```

### 4. 集成检测验证器

将所有检测器集成在一起，提供完整的面部特征实时检测功能。

```python
def visualize_integrated_results(image, 
                               left_eye_region, right_eye_region, 
                               left_eye_state, right_eye_state,
                               left_eye_conf, right_eye_conf,
                               mouth_region, yawn_state, yawn_conf,
                               roll, pitch, yaw,
                               face_landmarks=None):
    """整合可视化结果"""
    h, w, _ = image.shape
    
    # 1. 绘制关键点和区域
    if face_landmarks:
        # 绘制眼睛和嘴部关键点
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            pos = (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            cv2.circle(image, pos, 1, (0, 255, 0), -1)
        
        for idx in MOUTH_INDICES:
            pos = (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            cv2.circle(image, pos, 1, (0, 165, 255), -1)
    
    # 2. 绘制区域模块 - 移到左上角
    region_size = 80
    region_padding = 5
    start_x = 10
    start_y = 10
    
    # 统一处理区域显示
    def display_region(region, x, y, size, state, color=None):
        if region is not None and region.size > 0:
            display = cv2.resize(region, (size, size))
            image[y:y+size, x:x+size] = display
            if color is None:
                color = (0, 255, 0)  # 默认绿色
            cv2.rectangle(image, (x, y), (x+size, y+size), color, 2)
    
    # 左眼区域
    eye_color = lambda state: (0, 255, 0) if state == "open_eyes" else (0, 0, 255)
    display_region(left_eye_region, start_x, start_y, region_size, 
                  left_eye_state, eye_color(left_eye_state))
    
    # 右眼区域
    start_x += region_size + region_padding
    display_region(right_eye_region, start_x, start_y, region_size, 
                  right_eye_state, eye_color(right_eye_state))
    
    # 嘴部区域
    start_x += region_size + region_padding
    yawn_color = (0, 255, 0) if yawn_state == "no_yawn" else (0, 0, 255)
    display_region(mouth_region, start_x, start_y, region_size, 
                  yawn_state, yawn_color)
    
    # 3. 在左下角创建半透明背景区域显示状态信息
    text_area_height = 100
    text_area_width = 400
    text_area_x = 10
    text_area_y = h - text_area_height - 10
    
    # 创建半透明覆盖层
    overlay = image.copy()
    cv2.rectangle(overlay, (text_area_x, text_area_y), 
                 (text_area_x + text_area_width, text_area_y + text_area_height), 
                 (0, 0, 0), -1)  # 黑色背景
    
    # 设置半透明度
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 4. 在半透明区域上添加文本状态
    # 确定眼睛状态和颜色
    if left_eye_state == "closed_eyes" and right_eye_state == "closed_eyes":
        eye_status = "BOTH EYES CLOSED"
        eye_color = (0, 0, 255)  # 红色
    elif left_eye_state == "closed_eyes" or right_eye_state == "closed_eyes":
        eye_status = "ONE EYE CLOSED"
        eye_color = (0, 165, 255)  # 橙色
    else:
        eye_status = "EYES OPEN"
        eye_color = (0, 255, 0)  # 绿色
    
    # 确定嘴部状态和颜色
    if yawn_state == "yawn":
        yawn_text = "YAWNING DETECTED"
        yawn_color = (0, 0, 255)  # 红色
    else:
        yawn_text = "NO YAWN"
        yawn_color = (0, 255, 0)  # 绿色
    
    # 绘制文本信息
    line_height = 25
    cv2.putText(image, f"Eyes: {eye_status}", 
               (text_area_x + 10, text_area_y + line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
    
    cv2.putText(image, f"Mouth: {yawn_text}", 
               (text_area_x + 10, text_area_y + line_height*2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, yawn_color, 2)
    
    # 欧拉角显示
    cv2.putText(image, f"Roll : {roll:.1f}  Pitch : {pitch:.1f}  Yaw : {yaw:.1f}",
               (text_area_x + 10, text_area_y + line_height*3), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image
```

## 四、疲劳判断模块

### 1. 疲劳等级定义

```python
class FatigueLevel(Enum):
    """疲劳等级枚举"""
    NORMAL = 0       # 正常
    MILD = 1         # 轻度疲劳
    MODERATE = 2     # 中度疲劳
    SEVERE = 3       # 重度疲劳
    DANGEROUS = 4    # 危险驾驶
```

### 2. 疲劳分析器

基于检测到的各项特征，判断驾驶员的疲劳程度。系统使用滑动窗口来计算每分钟眨眼频率和每小时打哈欠频率，通过加权评分系统对驾驶员的疲劳程度进行打分。

```python
class FatigueAnalyzer:
    """疲劳分析器，用于评估驾驶员的疲劳程度"""
    
    def __init__(self, window_size_minutes=5):
        # 滑动窗口相关变量
        self.window_size_minutes = window_size_minutes
        self.initial_blink_count = 0
        self.initial_yawn_count = 0
        self.initial_nod_count = 0
        
        # 状态变量
        self.head_down = False
        self.eyes_closed = False
        self.current_fatigue_level = FatigueLevel.NORMAL
        self.current_fatigue_score = 0
        
        # 阈值设置
        self.blink_thresholds = {
            "normal": (10, 15),      # 正常：10-15次/分钟
            "mild": (15, 25),        # 轻度疲劳：15-25次/分钟
            "moderate": (25, 30),    # 中度疲劳：25-30次/分钟
            "severe": 30             # 重度疲劳：>30次/分钟
        }
        
        # 哈欠频率阈值（次/小时）
        self.yawn_thresholds = {
            "normal": (0.2, 0.4),    # 正常：0.2-0.4次/小时
            "mild": (1, 2),          # 轻度疲劳：1-2次/小时
            "moderate": (3, 5),      # 中度疲劳：3-5次/小时
            "severe": 5              # 重度疲劳：>5次/小时
        }
```

### 3. 疲劳数据记录器

记录和可视化疲劳相关指标，包括眨眼次数、哈欠次数、头部姿态等。数据记录模块会记录每一秒的驾驶员状态，为后续的疲劳分析和预测提供数据支持。

```python
class FatigueDataRecorder:
    """疲劳数据记录器，用于记录和可视化眼睛状态、哈欠状态和头部姿态数据"""
    
    def __init__(self, max_points=300):
        # 多线程处理相关
        self.lock = threading.Lock()
        self.raw_data_queue = queue.Queue()        # 原始数据队列
        self.processed_data_queue = queue.Queue()  # 处理后数据队列
        self.display_data_queue = queue.Queue()    # 显示数据队列
        
        # 数据存储
        self.timestamps = []
        self.eye_states = []  # 0: Eyes open, 1: One eye closed, 2: Both eyes closed
        self.yawn_states = []  # 0: No yawn, 1: Yawning
        self.roll_angles = []
        self.pitch_angles = []
        self.yaw_angles = []
        
        # 疲劳分析器
        self.fatigue_analyzer = FatigueAnalyzer(window_size_minutes=5)
```

### 4. 疲劳评估算法

系统使用加权评分机制，综合考虑眨眼频率、哈欠频率、头部姿态等多个因素，对驾驶员的疲劳程度进行综合评估。

```python
def _calculate_fatigue_level(self, blink_count, yawn_count, eyes_closed, head_down, recent_nod):
    """
    根据各项指标计算疲劳等级
    """
    # 计算眨眼频率（次/分钟）
    current_time = time.time()
    elapsed_minutes = max(1.0, (current_time - self.window_start_time) / 60)
    
    blink_increment = blink_count - self.initial_blink_count
    self.current_blink_rate = blink_increment / elapsed_minutes
    
    # 计算哈欠频率（次/小时）
    self.current_yawn_rate = yawn_count - self.initial_yawn_count
    
    # 特殊情况判断：危险驾驶
    if eyes_closed and head_down:
        self.current_fatigue_level = FatigueLevel.DANGEROUS
        self.current_fatigue_score = 100
        return
    
    # 正常疲劳评分计算
    # ...
```

## 五、疲劳预测系统

系统实现了对检测数据的记录功能，构建了一个GRU时序分析模型。当驾驶数据足够时，可以使用这些驾驶数据来训练GRU预测模型，对驾驶员的未来疲劳状态进行预测。

### 1. GRU模型定义

```python
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
        # 初始化隐藏状态
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU层前向传播
        out, h_out = self.gru(x, h)
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        return out, h_out
```

### 2. 疲劳预测系统实现

预测系统使用眨眼率、哈欠率、疲劳得分等多个特征作为输入，预测未来一段时间内的疲劳状态。

```python
class FatiguePredictionSystem:
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, learning_rate=0.001, 
                 batch_size=32, sequence_length=60, prediction_horizon=20,
                 samples_per_hour=60):
        """
        初始化疲劳预测系统
        
        参数:
        - input_size: 输入特征维度（默认为5个特征）
        - sequence_length: 输入序列长度（代表1小时的数据点数量）
        - prediction_horizon: 预测时间范围（20分钟后）
        """
        self.model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
```

### 3. 滑动窗口预测机制

```python
def train_with_sliding_window(self, csv_file_path, num_epochs=2, window_epochs=1,
                          validation_ratio=0.2, patience=3, min_samples_per_window=10, 
                          window_step_size=None):
    """
    使用滑动窗口方法训练模型
    
    参数:
    - csv_file_path: CSV文件或目录路径
    - num_epochs: 总体训练轮数
    - window_epochs: 每个窗口的训练轮数
    - window_step_size: 窗口滑动步长
    """
    # 窗口化训练实现
    # ...
```

### 4. 预测系统特点

1. 使用滑动窗口预测未来疲劳状态
2. 基于多个疲劳特征进行综合分析（眨眼率、哈欠率、疲劳得分等）
3. 支持动态调整样本窗口大小
4. 提供可视化预测结果

## 六、系统工作流程

1. **数据采集**：通过摄像头采集驾驶员面部视频流
2. **特征提取**：
   - 使用MediaPipe提取面部关键点
   - 检测眼睛开闭状态
   - 检测哈欠状态
   - 计算头部姿态角度
3. **疲劳分析**：
   - 计算眨眼频率
   - 分析哈欠频率
   - 检测头部低垂和点头状态
   - 综合评估疲劳程度
4. **数据记录**：
   - 实时记录所有疲劳指标
   - 生成CSV数据文件
5. **疲劳预测**：
   - 基于历史数据训练GRU模型
   - 预测未来疲劳趋势
6. **警报系统**：
   - 根据当前疲劳级别和预测结果生成警报

![系统流程图](系统流程图.png)

## 七、系统集成示例

```python
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    eyes_model = load_eyes_model(EYES_MODEL_PATH, device)
    yawn_model = load_yawn_model(YAWN_MODEL_PATH, device)
    
    # 初始化数据记录器和分析器
    data_recorder = FatigueDataRecorder()
    data_recorder.start()
    
    # 初始化MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            # 读取和处理视频帧
            # 检测面部特征
            # 分析疲劳状态
            # 可视化结果
            # ...
```

## 八、总结

本系统通过先进的深度学习技术实现了对驾驶员疲劳状态的实时监测和预测。系统使用摄像头捕捉实时视频流，通过Mediapipe关键点检测技术对眼部和嘴部进行裁剪，将裁剪后的图像数据输入到训练好的ResNet-18模型中进行识别。系统集成了眼睛状态检测、哈欠检测、头部姿态分析等多种特征提取方法，并通过疲劳分析器对这些特征进行综合评估，最终通过GRU神经网络实现了对未来疲劳状态的预测。

系统的优点包括：
1. 多特征融合，全面评估疲劳状态
2. 实时性能好，能够在普通硬件上流畅运行
3. 提供预测功能，可以提前预警危险驾驶状态
4. 模块化设计，便于维护和扩展

该系统可以广泛应用于驾驶安全监控、长途货运司机管理、智能座舱系统等场景，有效减少因疲劳驾驶导致的交通事故。 