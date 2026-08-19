import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 添加父目录到路径，以便导入训练模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train.eyes_resnet_train import build_model

# MediaPipe Face Mesh 初始化
mp_face_mesh = mp.solutions.face_mesh

# 左眼和右眼的关键点索引
LEFT_EYE_INDICES = [35, 70, 52, 55, 188, 120, 117]
RIGHT_EYE_INDICES = [285, 295, 300, 265, 346, 349, 412]

class EyeImageDataset(Dataset):
    """眼睛图像数据集类"""
    def __init__(self, data_dir, transform=None):
        """
        参数:
            data_dir (str): 数据集目录路径，其中应包含'closed_eyes'和'open_eyes'子目录
            transform (callable, optional): 应用于图像的变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = ["closed_eyes", "open_eyes"]
        self.samples = []
        
        # 遍历类别目录并收集样本
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 目录 {class_dir} 不存在")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def load_model(model_path, device):
    """加载训练好的模型"""
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transforms():
    """获取与训练时相同的数据变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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

def predict_eye_state(model, eye_region, transform, device, class_names, threshold=0.6):
    """直接预测眼睛状态"""
    if eye_region is None or eye_region.size == 0:
        return "Unknown", 0.0
    
    # 转换为PIL图像
    try:
        if isinstance(eye_region, str):  # 如果是图像路径
            pil_image = Image.open(eye_region).convert('RGB')
        elif isinstance(eye_region, np.ndarray):  # 如果是图像数组
            pil_image = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
        else:
            return "Unknown", 0.0
            
        # 应用变换并转换为模型输入
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            # 应用阈值
            if prediction.item() == 0 and confidence.item() < threshold:  # closed_eyes
                prediction = 1  # open_eyes
        
        return class_names[prediction.item()], confidence.item()
    except Exception as e:
        print(f"预测眼睛状态时出错: {e}")
        return "Unknown", 0.0

def evaluate_model_on_dataset(model, data_loader, device, class_names):
    """在数据集上评估模型性能"""
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="评估模型"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\n模型评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    print("\n混淆矩阵:")
    print(conf_matrix)
    print("\n类别对应:")
    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def evaluate_model_on_image_folders(model, image_dir, device, class_names, threshold=0.6):
    """直接评估模型在图像文件夹上的性能，无需面部检测"""
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 {image_dir} 不存在")
        return None
    
    # 获取数据变换
    transform = get_transforms()
    
    # 搜索子文件夹以获取所有图像
    image_folders = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_folders.append(root)
                break
    
    image_folders = list(set(image_folders))  # 去重
    
    if not image_folders:
        print(f"错误: 在 {image_dir} 中未找到有效的图像文件夹")
        return None
    
    # 评估结果
    folder_results = {}
    
    for folder in image_folders:
        folder_name = os.path.basename(folder)
        print(f"\n处理文件夹: {folder_name}")
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 初始化计数器
        open_eyes = 0
        closed_eyes = 0
        unknown = 0
        
        for image_file in tqdm(image_files, desc=f"处理 {folder_name}"):
            image_path = os.path.join(folder, image_file)
            
            # 直接预测图像
            eye_state, confidence = predict_eye_state(
                model, image_path, transform, device, class_names, threshold
            )
            
            if eye_state == "open_eyes":
                open_eyes += 1
            elif eye_state == "closed_eyes":
                closed_eyes += 1
            else:
                unknown += 1
        
        # 计算结果
        total_images = len(image_files)
        open_ratio = open_eyes / total_images if total_images > 0 else 0
        closed_ratio = closed_eyes / total_images if total_images > 0 else 0
        
        folder_results[folder_name] = {
            'total_images': total_images,
            'open_eyes': open_eyes,
            'closed_eyes': closed_eyes,
            'unknown': unknown,
            'open_ratio': open_ratio,
            'closed_ratio': closed_ratio
        }
        
        print(f"\n文件夹 {folder_name} 统计信息:")
        print(f"总图像数: {total_images}")
        print(f"睁眼图像数: {open_eyes}")
        print(f"闭眼图像数: {closed_eyes}")
        print(f"未知状态数: {unknown}")
        print(f"睁眼比例: {open_ratio:.4f}")
        print(f"闭眼比例: {closed_ratio:.4f}")
    
    # 计算整体统计数据
    total_images = sum(result['total_images'] for result in folder_results.values())
    total_open = sum(result['open_eyes'] for result in folder_results.values())
    total_closed = sum(result['closed_eyes'] for result in folder_results.values())
    total_unknown = sum(result['unknown'] for result in folder_results.values())
    
    overall_open_ratio = total_open / total_images if total_images > 0 else 0
    overall_closed_ratio = total_closed / total_images if total_images > 0 else 0
    
    print("\n整体统计信息:")
    print(f"总图像数: {total_images}")
    print(f"睁眼图像总数: {total_open}")
    print(f"闭眼图像总数: {total_closed}")
    print(f"未知状态总数: {total_unknown}")
    print(f"整体睁眼比例: {overall_open_ratio:.4f}")
    print(f"整体闭眼比例: {overall_closed_ratio:.4f}")
    
    return folder_results

def evaluate_model_on_video_dataset(model, video_dir, device, class_names, threshold=0.6, direct_detection=False):
    """评估模型在视频数据集上的性能，支持无面部检测时的直接检测"""
    # 检查目录是否存在
    if not os.path.exists(video_dir):
        print(f"错误: 视频目录 {video_dir} 不存在")
        return None
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"错误: 在 {video_dir} 中未找到视频文件")
        return None
    
    # 获取数据变换
    transform = get_transforms()
    
    # 初始化MediaPipe Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 初始化统计数据
    total_frames = 0
    detected_faces = 0
    total_direct_detections = 0
    
    # 视频评估结果
    video_results = {}
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\n处理视频: {video_file}")
        
        # 初始化视频捕获
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            continue
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"总帧数: {frame_count}")
        
        # 初始化计数器
        video_frames = 0
        video_detected_faces = 0
        video_direct_detections = 0
        left_eye_open = 0
        right_eye_open = 0
        left_eye_closed = 0
        right_eye_closed = 0
        
        try:
            with tqdm(total=frame_count, desc=f"处理 {video_file}") as pbar:
                while cap.isOpened():
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    video_frames += 1
                    total_frames += 1
                    
                    # 转换为RGB进行MediaPipe处理
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 禁用写入保护以提高性能
                    rgb_frame.flags.writeable = False
                    results = face_mesh.process(rgb_frame)
                    rgb_frame.flags.writeable = True
                    
                    # 检查是否检测到面部关键点
                    if results.multi_face_landmarks:
                        video_detected_faces += 1
                        detected_faces += 1
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # 提取左右眼区域
                        left_eye_region, right_eye_region = extract_eye_regions(
                            frame, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
                        )
                        
                        # 预测左右眼状态
                        if left_eye_region is not None and left_eye_region.size > 0:
                            pil_image = Image.fromarray(cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2RGB))
                            input_tensor = transform(pil_image).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                confidence, prediction = torch.max(probabilities, 1)
                                
                                # 应用阈值
                                if prediction.item() == 0 and confidence.item() < threshold:  # closed_eyes
                                    prediction = 1  # open_eyes
                                
                                if prediction.item() == 1:  # open_eyes
                                    left_eye_open += 1
                                else:  # closed_eyes
                                    left_eye_closed += 1
                        
                        if right_eye_region is not None and right_eye_region.size > 0:
                            pil_image = Image.fromarray(cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2RGB))
                            input_tensor = transform(pil_image).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                confidence, prediction = torch.max(probabilities, 1)
                                
                                # 应用阈值
                                if prediction.item() == 0 and confidence.item() < threshold:  # closed_eyes
                                    prediction = 1  # open_eyes
                                
                                if prediction.item() == 1:  # open_eyes
                                    right_eye_open += 1
                                else:  # closed_eyes
                                    right_eye_closed += 1
                    elif direct_detection:
                        # 如果未检测到面部但启用了直接检测，则将整个帧作为输入
                        video_direct_detections += 1
                        total_direct_detections += 1
                        
                        # 直接在全帧上进行预测
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        input_tensor = transform(pil_image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, prediction = torch.max(probabilities, 1)
                            
                            # 为了统计目的，我们将结果分配给左眼和右眼
                            if prediction.item() == 1:  # open_eyes
                                left_eye_open += 1
                                right_eye_open += 1
                            else:  # closed_eyes
                                left_eye_closed += 1
                                right_eye_closed += 1
                    
                    pbar.update(1)
        
        finally:
            cap.release()
        
        # 计算视频统计数据
        detected_frames = video_detected_faces + video_direct_detections
        face_detection_rate = video_detected_faces / video_frames if video_frames > 0 else 0
        direct_detection_rate = video_direct_detections / video_frames if video_frames > 0 else 0
        left_eye_open_rate = left_eye_open / detected_frames if detected_frames > 0 else 0
        right_eye_open_rate = right_eye_open / detected_frames if detected_frames > 0 else 0
        
        video_results[video_file] = {
            'frames': video_frames,
            'detected_faces': video_detected_faces,
            'direct_detections': video_direct_detections,
            'total_detections': detected_frames,
            'face_detection_rate': face_detection_rate,
            'direct_detection_rate': direct_detection_rate,
            'left_eye_open': left_eye_open,
            'left_eye_closed': left_eye_closed,
            'right_eye_open': right_eye_open,
            'right_eye_closed': right_eye_closed,
            'left_eye_open_rate': left_eye_open_rate,
            'right_eye_open_rate': right_eye_open_rate
        }
        
        print(f"\n视频 {video_file} 统计信息:")
        print(f"总帧数: {video_frames}")
        print(f"检测到人脸的帧数: {video_detected_faces}")
        print(f"直接检测的帧数: {video_direct_detections}")
        print(f"总检测帧数: {detected_frames}")
        print(f"人脸检测率: {face_detection_rate:.4f}")
        print(f"直接检测率: {direct_detection_rate:.4f}")
        print(f"左眼睁开次数: {left_eye_open}")
        print(f"左眼闭合次数: {left_eye_closed}")
        print(f"右眼睁开次数: {right_eye_open}")
        print(f"右眼闭合次数: {right_eye_closed}")
        print(f"左眼睁开率: {left_eye_open_rate:.4f}")
        print(f"右眼睁开率: {right_eye_open_rate:.4f}")
    
    # 计算整体统计数据
    total_detected = detected_faces + total_direct_detections
    overall_face_detection_rate = detected_faces / total_frames if total_frames > 0 else 0
    overall_direct_detection_rate = total_direct_detections / total_frames if total_frames > 0 else 0
    
    print("\n整体统计信息:")
    print(f"总帧数: {total_frames}")
    print(f"检测到人脸的帧数: {detected_faces}")
    print(f"直接检测的帧数: {total_direct_detections}")
    print(f"总检测帧数: {total_detected}")
    print(f"整体人脸检测率: {overall_face_detection_rate:.4f}")
    print(f"整体直接检测率: {overall_direct_detection_rate:.4f}")
    
    return video_results

def main():
    parser = argparse.ArgumentParser(description='眼睛状态检测模型评估')
    parser.add_argument('--model_path', type=str, default="D:\Project\guaduation_project\models\eyes_resnet_18_l.pth",
                        help='模型文件路径')
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\woaiy\OneDrive\Desktop\dataset_new\train\eye",help='图像数据集目录路径')
    parser.add_argument('--video_dir', type=str, help='视频数据集目录路径')
    parser.add_argument('--image_folders_dir', type=str, help='直接评估的图像文件夹目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--threshold', type=float, default=0.9, help='闭眼识别阈值')
    parser.add_argument('--direct_detection', action='store_true', 
                        help='是否在未检测到人脸时对整个帧进行直接检测')
    
    args = parser.parse_args()
    
    # 检查是否提供了数据目录
    if not args.data_dir and not args.video_dir and not args.image_folders_dir:
        parser.error("必须提供 --data_dir, --video_dir 或 --image_folders_dir 参数中的一个")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    try:
        model = load_model(args.model_path, device)
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 类别名称
    class_names = ["closed_eyes", "open_eyes"]
    
    # 在图像数据集上评估
    if args.data_dir:
        # 获取数据变换
        transform = get_transforms()
        
        # 创建数据集和数据加载器
        test_dataset = EyeImageDataset(args.data_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 评估模型
        evaluate_model_on_dataset(model, test_loader, device, class_names)
    
    # 在视频数据集上评估
    if args.video_dir:
        evaluate_model_on_video_dataset(
            model, args.video_dir, device, class_names, 
            threshold=args.threshold, direct_detection=args.direct_detection
        )
    
    # 在图像文件夹上直接评估
    if args.image_folders_dir:
        evaluate_model_on_image_folders(model, args.image_folders_dir, device, class_names, threshold=args.threshold)

if __name__ == "__main__":
    main() 