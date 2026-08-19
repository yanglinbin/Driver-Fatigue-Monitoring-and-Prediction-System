import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
import cupy as cp

# 添加父目录到路径，以便导入训练模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入其他验证器模块中的函数
from src.validation.eye_detection_validator import (
    load_model as load_eyes_model,
    get_transforms,
    extract_eye_regions,
    predict_eye_state_with_threshold,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES
)

from src.validation.pose_detection_validation import (
    get_landmark_coords,
    fit_plane,
    compute_euler_angles,
    draw_axes,
    draw_plane,
    PLANE_IDX
)

from src.validation.yawn_detection_validator import (
    load_model as load_yawn_model,
    extract_mouth_region,
    predict_yawn_state,
    MOUTH_INDICES
)

# 导入数据记录器
from src.fatigue_judgment.fatigue_data_recorder import FatigueDataRecorder

# MediaPipe Face Mesh 初始化
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 模型路径
EYES_MODEL_PATH = "D:\Project\guaduation_project\models\eyes_resnet_18_l.pth"
YAWN_MODEL_PATH = "D:\Project\guaduation_project\models\yawn_resnet_18.pth"

# ========================
# 可视化函数
# ========================
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
    cv2.putText(image, f"L:{left_eye_conf:.2f}", (start_x, start_y+region_size+15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 右眼区域
    start_x += region_size + region_padding
    display_region(right_eye_region, start_x, start_y, region_size, 
                  right_eye_state, eye_color(right_eye_state))
    cv2.putText(image, f"R:{right_eye_conf:.2f}", (start_x, start_y+region_size+15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 嘴部区域
    start_x += region_size + region_padding
    yawn_color = (0, 255, 0) if yawn_state == "no_yawn" else (0, 0, 255)
    display_region(mouth_region, start_x, start_y, region_size, 
                  yawn_state, yawn_color)
    cv2.putText(image, f"Y:{yawn_conf:.2f}", (start_x, start_y+region_size+15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 3. 在左下角创建半透明背景区域
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

# ========================
# 主函数
# ========================
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载眼睛检测模型
    try:
        eyes_model = load_eyes_model(EYES_MODEL_PATH, device)
        print("Eye detection model loaded successfully")
    except Exception as e:
        print(f"Failed to load eye detection model: {e}")
        return
    
    # 加载哈欠检测模型
    try:
        yawn_model = load_yawn_model(YAWN_MODEL_PATH, device)
        print("Yawn detection model loaded successfully")
    except Exception as e:
        print(f"Failed to load yawn detection model: {e}")
        return
    
    # 获取数据变换
    transform = get_transforms()
    
    # 定义类别名称
    eyes_class_names = ["closed_eyes", "open_eyes"]
    yawn_class_names = ["no_yawn", "yawn"]
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(r"D:\Project\数据集\video_\111.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 验证实际分辨率
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution set to: {actual_width}x{actual_height}")
    
    # 设置目标帧率
    TARGET_FPS = 30
    frame_time = 1.0 / TARGET_FPS
    
    # 初始化历史状态队列
    history_length = 5
    left_eye_history = deque([0] * history_length, maxlen=history_length)
    right_eye_history = deque([0] * history_length, maxlen=history_length)
    yawn_history = deque([0] * history_length, maxlen=history_length)
    
    # 初始化欧拉角历史数据
    roll_history = deque([0.0] * history_length, maxlen=history_length)
    pitch_history = deque([0.0] * history_length, maxlen=history_length)
    yaw_history = deque([0.0] * history_length, maxlen=history_length)
    
    # 初始化数据记录器
    data_recorder = FatigueDataRecorder()
    data_recorder.start()
    
    # 初始化键盘记录标志
    is_recording = False
    
    # 定义头部快速转动的阈值（角度/帧）
    head_movement_threshold = 3.0
    
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
                
                # 获取图像尺寸
                h, w, _ = frame.shape
                
                # 转换为RGB进行MediaPipe处理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 设置为只读模式以提高性能
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                # 初始化变量
                left_eye_region = right_eye_region = mouth_region = None
                left_eye_state = right_eye_state = "Unknown"
                yawn_state = "Unknown"
                left_eye_conf = right_eye_conf = yawn_conf = 0.0
                roll = pitch = yaw = 0.0
                
                # 检查是否检测到面部关键点
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # 1. 提取眼睛区域和嘴部区域
                    left_eye_region, right_eye_region = extract_eye_regions(
                        frame, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
                    )
                    
                    mouth_region = extract_mouth_region(
                        frame, face_landmarks, MOUTH_INDICES
                    )
                    
                    # 2. 姿态估计
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
                    
                    # 更新欧拉角历史
                    roll_history.append(roll)
                    pitch_history.append(pitch)
                    yaw_history.append(yaw)
                    
                    # 检测头部快速转动
                    if len(roll_history) > 1:
                        roll_change = abs(roll_history[-1] - roll_history[-2])
                        pitch_change = abs(pitch_history[-1] - pitch_history[-2])
                        yaw_change = abs(yaw_history[-1] - yaw_history[-2])
                        
                        if (roll_change > head_movement_threshold or 
                            pitch_change > head_movement_threshold or 
                            yaw_change > head_movement_threshold):
                            print(f"Head movement detected: Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}")
                            
                    # 3. 眼睛状态检测
                    if left_eye_region is not None and left_eye_region.size > 0:
                        left_eye_state, left_eye_conf = predict_eye_state_with_threshold(
                            eyes_model, left_eye_region, transform, device, eyes_class_names, threshold=0.6
                        )
                        left_eye_history.append(1 if left_eye_state == "open_eyes" else 0)
                    
                    if right_eye_region is not None and right_eye_region.size > 0:
                        right_eye_state, right_eye_conf = predict_eye_state_with_threshold(
                            eyes_model, right_eye_region, transform, device, eyes_class_names, threshold=0.6
                        )
                        right_eye_history.append(1 if right_eye_state == "open_eyes" else 0)
                    
                    # 状态平滑处理
                    if len(left_eye_history) > 0:
                        left_open_ratio = sum(left_eye_history) / len(left_eye_history)
                        left_eye_state = "open_eyes" if left_open_ratio > 0.7 else "closed_eyes"
                    
                    if len(right_eye_history) > 0:
                        right_open_ratio = sum(right_eye_history) / len(right_eye_history)
                        right_eye_state = "open_eyes" if right_open_ratio > 0.7 else "closed_eyes"
                    
                    # 4. 哈欠检测
                    if mouth_region is not None and mouth_region.size > 0:
                        yawn_state, yawn_conf = predict_yawn_state(
                            yawn_model, mouth_region, transform, device, yawn_class_names
                        )
                        yawn_history.append(1 if yawn_state == "yawn" else 0)
                    
                    # 平滑哈欠状态
                    if len(yawn_history) > 0:
                        yawn_ratio = sum(yawn_history) / len(yawn_history)
                        yawn_state = "yawn" if yawn_ratio > 0.6 else "no_yawn"
                    
                    # 5. 准备数据记录
                    eye_status_text = "EYES OPEN"
                    if left_eye_state == "closed_eyes" and right_eye_state == "closed_eyes":
                        eye_status_text = "BOTH EYES CLOSED"
                    elif left_eye_state == "closed_eyes" or right_eye_state == "closed_eyes":
                        eye_status_text = "ONE EYE CLOSED"
                    
                    yawn_status_text = "YAWNING DETECTED" if yawn_state == "yawn" else "NO YAWN"
                    
                    # 记录疲劳数据（当录制模式开启时）
                    if is_recording:
                        data_recorder.add_data_point(eye_status_text, yawn_status_text, roll, pitch, yaw)
                    
                    # 可视化结果
                    frame = visualize_integrated_results(
                        frame, left_eye_region, right_eye_region,
                        left_eye_state, right_eye_state,
                        left_eye_conf, right_eye_conf,
                        mouth_region, yawn_state, yawn_conf,
                        roll, pitch, yaw,
                        face_landmarks
                    )
                else:
                    # 如果没有检测到面部，显示消息
                    cv2.putText(frame, "No Face Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 计算FPS并显示
                elapsed_time = time.time() - start_time
                fps = 1.0 / max(0.001, elapsed_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 录制状态显示
                if is_recording:
                    cv2.putText(frame, "Recording Data", (frame.shape[1] - 250, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示帧
                cv2.imshow("Integrated Fatigue Detection", frame)
                
                # 计算需要等待的时间以维持目标帧率
                elapsed_time = time.time() - start_time
                wait_time = max(1, int((frame_time - elapsed_time) * 1000))
                
                # 键盘控制
                key = cv2.waitKey(wait_time) & 0xFF
                if key == 27:  # ESC键退出
                    break
                elif key == ord('r'):  # 'r'键切换录制状态
                    is_recording = not is_recording
                    if is_recording:
                        print("Started recording fatigue data")
                    else:
                        print("Stopped recording fatigue data")
                
        finally:
            # 停止数据记录
            data_recorder.stop()
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 