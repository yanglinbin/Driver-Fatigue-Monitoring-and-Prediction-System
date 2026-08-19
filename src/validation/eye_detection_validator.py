import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image

# 添加父目录到路径，以便导入训练模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train.eyes_resnet_train import build_model

# MediaPipe Face Mesh 初始化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 左眼和右眼的关键点索引
LEFT_EYE_INDICES = [35, 70, 52, 55, 188, 120, 117]
RIGHT_EYE_INDICES = [285, 295, 300, 265, 346, 349, 412]

# 模型路径
MODEL_PATH = "D:\Project\guaduation_project\models\eyes_resnet_18_l.pth"

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

def predict_eye_state(model, eye_region, transform, device, class_names):
    """预测眼睛状态"""
    if eye_region is None or eye_region.size == 0:
        return "Unknown", 0.0
    
    # 转换为PIL图像
    try:
        pil_image = Image.fromarray(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB))
        # 应用变换并转换为模型输入
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(input_tensor)
            # 使用 softmax 获取概率分布
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        return class_names[prediction.item()], confidence.item()
    except Exception as e:
        print(f"Error in eye state prediction: {e}")
        return "Unknown", 0.0

def predict_eye_state_with_threshold(model, eye_region, transform, device, class_names, threshold=0.50):
    """使用更低的置信度阈值的眼睛状态预测函数"""
    pred_class, confidence = predict_eye_state(model, eye_region, transform, device, class_names)
    
    # 降低阈值到0.55，因为标签平滑导致模型输出的概率值更温和
    if pred_class == "closed_eyes" and confidence < threshold:
        return "open_eyes", confidence
    return pred_class, confidence

def visualize_results(image, left_eye_region, right_eye_region, left_eye_state, right_eye_state, 
                      left_eye_conf, right_eye_conf, face_landmarks=None):
    """在图像上可视化结果"""
    h, w, _ = image.shape
    
    # 如果有面部关键点，绘制左右眼区域
    if face_landmarks:
        # 绘制左眼和右眼关键点
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            pos = (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            cv2.circle(image, pos, 2, (0, 255, 0), -1)
    
    # 在图像上添加左右眼状态
    cv2.putText(image, f"Left Eye: {left_eye_state} ({left_eye_conf:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Right Eye: {right_eye_state} ({right_eye_conf:.2f})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 确定整体眼睛状态及显示颜色
    if left_eye_state == "closed_eyes" and right_eye_state == "closed_eyes":
        overall_state = "BOTH EYES CLOSED"
        color = (0, 0, 255)  # 红色
    elif left_eye_state == "closed_eyes" or right_eye_state == "closed_eyes":
        overall_state = "ONE EYE CLOSED"
        color = (0, 165, 255)  # 橙色
    else:
        overall_state = "EYES OPEN"
        color = (0, 255, 0)  # 绿色
    
    cv2.putText(image, overall_state, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # 显示左右眼区域
    display_size = 100  # 显示区域大小
    
    # 显示左眼区域
    if left_eye_region is not None and left_eye_region.size > 0:
        left_eye_display = cv2.resize(left_eye_region, (display_size, display_size))
        image[120:120+display_size, 10:10+display_size] = left_eye_display
        cv2.rectangle(image, (10, 120), (10+display_size, 120+display_size), 
                     (0, 255, 0) if left_eye_state == "open_eyes" else (0, 0, 255), 2)
    
    # 显示右眼区域
    if right_eye_region is not None and right_eye_region.size > 0:
        right_eye_display = cv2.resize(right_eye_region, (display_size, display_size))
        image[120:120+display_size, 120:120+display_size] = right_eye_display
        cv2.rectangle(image, (120, 120), (120+display_size, 120+display_size), 
                     (0, 255, 0) if right_eye_state == "open_eyes" else (0, 0, 255), 2)
    
    return image

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    try:
        model = load_model(MODEL_PATH, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 获取数据变换
    transform = get_transforms()
    
    # 定义类别名称（与训练数据一致）
    class_names = ["closed_eyes", "open_eyes"]
    
    # 初始化视频捕获
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"D:\Project\数据集\video_\1-FemaleNoGlasses-Normal.avi")
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # 设置视频属性 - 固定帧率为30fps
    target_fps = 30
    # 尝试设置摄像头或视频的FPS
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    # 获取实际帧率
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"设置的目标帧率: {target_fps}, 实际帧率: {actual_fps}")
    
    # 计算帧间等待时间(毫秒)
    frame_delay = int(1000 / target_fps)
    
    # 初始化历史状态队列进行平滑
    history_length = 5
    left_eye_history = deque([0] * history_length, maxlen=history_length)
    right_eye_history = deque([0] * history_length, maxlen=history_length)
    
    # 初始化MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        try:
            while cap.isOpened():
                start_time = time.time()
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # 翻转图像以获得自拍视图
                frame = cv2.flip(frame, 1)
                
                # 转换为RGB进行MediaPipe处理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 禁用写入保护以提高性能
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                # 初始化眼睛区域和状态
                left_eye_region = right_eye_region = None
                left_eye_state = right_eye_state = "Unknown"
                left_eye_conf = right_eye_conf = 0.0
                
                # 检查是否检测到面部关键点
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # 提取左右眼区域
                    left_eye_region, right_eye_region = extract_eye_regions(
                        frame, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
                    )
                    
                    # 预测左右眼状态
                    if left_eye_region is not None and left_eye_region.size > 0:
                        left_eye_state, left_eye_conf = predict_eye_state_with_threshold(
                            model, left_eye_region, transform, device, class_names, threshold=0.6
                        )
                        # 更新历史状态
                        left_eye_history.append(1 if left_eye_state == "open_eyes" else 0)
                    
                    if right_eye_region is not None and right_eye_region.size > 0:
                        right_eye_state, right_eye_conf = predict_eye_state_with_threshold(
                            model, right_eye_region, transform, device, class_names, threshold=0.6
                        )
                        # 更新历史状态
                        right_eye_history.append(1 if right_eye_state == "open_eyes" else 0)
                    
                    # 状态平滑处理
                    left_open_ratio = sum(left_eye_history) / len(left_eye_history)
                    right_open_ratio = sum(right_eye_history) / len(right_eye_history)
                    
                    # 根据平滑结果确定最终状态
                    left_eye_state = "open_eyes" if left_open_ratio > 0.5 else "closed_eyes"
                    right_eye_state = "open_eyes" if right_open_ratio > 0.5 else "closed_eyes"
                    
                    # 可视化结果
                    frame = visualize_results(
                        frame, left_eye_region, right_eye_region,
                        left_eye_state, right_eye_state,
                        left_eye_conf, right_eye_conf,
                        face_landmarks
                    )
                else:
                    # 如果没有检测到面部，显示消息
                    cv2.putText(frame, "No Face Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 计算并显示帧率
                elapsed_time = time.time() - start_time
                actual_fps = 1.0 / max(0.001, elapsed_time)
                fps_text = f"FPS: {actual_fps:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow("Eye State Detection", frame)
                
                # 计算需要等待的时间以维持固定帧率
                processing_time = time.time() - start_time
                wait_time = max(1, int(frame_delay - processing_time * 1000))  # 确保至少等待1ms
                
                # 按 Esc 键退出
                if cv2.waitKey(wait_time) & 0xFF == 27:
                    break
        
        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 