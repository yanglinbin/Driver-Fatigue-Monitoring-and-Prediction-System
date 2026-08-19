import os
import sys
import time

import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image

# 添加父目录到路径，以便导入训练模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train.yawn_resnet_train import build_model

# MediaPipe Face Mesh 初始化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 嘴巴区域的关键点索引
MOUTH_INDICES = [57, 165, 164, 391, 287, 406, 18, 182, 57]

# 模型路径
MODEL_PATH = r"D:\Project\guaduation_project\src\train\models\yawn_resnet\new\best_model.pth"

def load_model(model_path, device):
    """加载训练好的模型"""
    model = build_model(num_classes=2, dropout_rate=0.5)
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

def visualize_results(image, mouth_region, yawn_state, yawn_conf, face_landmarks=None):
    """在图像上可视化结果"""
    h, w, _ = image.shape
    
    # 如果有面部关键点，绘制嘴部区域
    if face_landmarks:
        # 绘制嘴部关键点
        for idx in MOUTH_INDICES:
            pos = (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            cv2.circle(image, pos, 2, (0, 255, 0), -1)
    
    # 在图像上添加嘴部状态
    cv2.putText(image, f"Yawn State: {yawn_state} ({yawn_conf:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 根据是否打哈欠设置状态颜色
    if yawn_state == "yawn":
        status_text = "YAWNING DETECTED"
        status_color = (0, 0, 255)  # 红色
    elif yawn_state == "no_yawn":
        status_text = "NO YAWN"
        status_color = (0, 255, 0)  # 绿色
    else:
        status_text = "UNKNOWN"
        status_color = (255, 0, 0)  # 蓝色
    
    cv2.putText(image, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # 显示嘴部区域
    if mouth_region is not None and mouth_region.size > 0:
        # 统一显示区域大小
        display_size = 150
        mouth_display = cv2.resize(mouth_region, (display_size, display_size))
        image[80:80+display_size, 10:10+display_size] = mouth_display
        cv2.rectangle(image, (10, 80), (10+display_size, 80+display_size), 
                     (0, 255, 0) if yawn_state == "no_yawn" else (0, 0, 255), 2)
    
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
    class_names = ["no_yawn", "yawn"]
    
    # 初始化视频捕获
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"D:\Project\数据集\video_\39-FemaleNoGlasses-Yawning.avi")
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
    
    # 平滑预测设置
    history_size = 5
    yawn_history = [0] * history_size
    yawn_threshold = 0.85  # 高阈值更倾向于no_yawn
    
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
                
                # 转换为RGB进行MediaPipe处理，并设置为只读模式提高性能
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                # 初始化嘴部区域和状态
                mouth_region = None
                yawn_state = "Unknown"
                yawn_conf = 0.0
                
                # 检查是否检测到面部关键点
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # 提取嘴部区域
                    mouth_region = extract_mouth_region(
                        frame, face_landmarks, MOUTH_INDICES
                    )
                    
                    # 预测是否打哈欠
                    if mouth_region is not None and mouth_region.size > 0:
                        yawn_state, yawn_conf = predict_yawn_state(
                            model, mouth_region, transform, device, class_names, threshold=yawn_threshold
                        )
                        
                        # 添加到历史记录以平滑预测
                        yawn_history.pop(0)
                        yawn_history.append(1 if yawn_state == "yawn" else 0)
                        
                        # 平滑预测 - 使用更高的阈值，更倾向于no_yawn
                        avg_yawn = sum(yawn_history) / len(yawn_history)
                        if avg_yawn > 0.6:  # 至少3/5的帧检测到打哈欠才认为是打哈欠
                            yawn_state = "yawn"
                        else:
                            yawn_state = "no_yawn"
                    
                    # 可视化结果
                    frame = visualize_results(
                        frame, mouth_region, yawn_state, yawn_conf, face_landmarks
                    )
                else:
                    # 如果没有检测到面部，显示消息
                    cv2.putText(frame, "No Face Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示帧率
                elapsed_time = time.time() - start_time
                actual_fps = 1.0 / max(0.001, elapsed_time)
                fps_text = f"FPS: {actual_fps:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow("Yawn Detection", frame)
                
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