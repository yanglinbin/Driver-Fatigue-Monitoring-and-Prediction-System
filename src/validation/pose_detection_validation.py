import math
import threading
import time
from queue import Queue
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import cupy as cp
from cupy.linalg import svd

# ========================
# 参数设置及初始化
# ========================
# 平面拟合点位更新为新的关键点索引
PLANE_IDX = [117, 346, 151, 9, 4, 23, 253]

# 全局变量，用于存储首次检测到的人脸局部坐标系作为基准
baseline_axes = None  # (base_x, base_y, base_z)

# FPS计算相关变量
fps = 0
frame_times = deque(maxlen=30)  # 存储最近30帧的处理时间

# 固定帧率设置
target_fps = 30
frame_delay = int(1000 / target_fps)

# 初始化 mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 用于绘制关键点
mp_drawing = mp.solutions.drawing_utils

# 用于多线程处理的队列（最大队列长度可根据需要调整）
frame_queue = Queue(maxsize=5)
stop_event = threading.Event()

# ========================
# 工具函数
# ========================
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

def draw_axes(image, center, x_axis, y_axis, z_axis, scale=30):
    """
    在图像上绘制坐标轴：
      - x_axis：红色（X）
      - y_axis：绿色（Y）
      - z_axis：蓝色（Z）
    """
    # 确保所有输入都是numpy数组
    center = cp.asnumpy(center) if isinstance(center, cp.ndarray) else center
    x_axis = cp.asnumpy(x_axis) if isinstance(x_axis, cp.ndarray) else x_axis
    y_axis = cp.asnumpy(y_axis) if isinstance(y_axis, cp.ndarray) else y_axis
    z_axis = cp.asnumpy(z_axis) if isinstance(z_axis, cp.ndarray) else z_axis
    
    origin = (int(center[0]), int(center[1]))
    x_end = (int(center[0] + x_axis[0] * scale), int(center[1] + x_axis[1] * scale))
    y_end = (int(center[0] + y_axis[0] * scale), int(center[1] + y_axis[1] * scale))
    z_end = (int(center[0] - z_axis[0] * scale), int(center[1] - z_axis[1] * scale))
    
    cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 2, tipLength=0.2)
    cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 2, tipLength=0.2)

def draw_plane(image, center, x_axis, y_axis, scale=60, color=(0, 255, 255), alpha=0.3):
    """
    在图像上绘制平面：利用 center、x_axis 和 y_axis 构造矩形区域并用半透明颜色填充
    """
    # 确保所有输入都是numpy数组
    center = cp.asnumpy(center) if isinstance(center, cp.ndarray) else center
    x_axis = cp.asnumpy(x_axis) if isinstance(x_axis, cp.ndarray) else x_axis
    y_axis = cp.asnumpy(y_axis) if isinstance(y_axis, cp.ndarray) else y_axis
    
    corners = np.array([
        center + x_axis * scale + y_axis * scale,
        center - x_axis * scale + y_axis * scale,
        center - x_axis * scale - y_axis * scale,
        center + x_axis * scale - y_axis * scale
    ])
    
    pts = np.array([[int(c[0]), int(c[1])] for c in corners], np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

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

def calculate_fps():
    """
    计算FPS：使用滑动窗口计算平均帧率
    """
    if len(frame_times) < 2:
        return 0
    
    # 计算时间差的倒数，得到每一帧的FPS
    fps_values = [1.0 / max(t2 - t1, 1e-6) for t1, t2 in zip(frame_times, list(frame_times)[1:])]
    # 返回平均FPS
    return sum(fps_values) / len(fps_values) if fps_values else 0

# ========================
# 多线程视频捕获
# ========================
def capture_thread_func(cap, queue, stop_event):
    """ 捕获视频帧并放入队列 """
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        # 翻转图像方便观察
        frame = cv2.flip(frame, 1)
        
        # 避免队列满的情况下阻塞
        if not queue.full():
            queue.put(frame)
        
        # 使用固定间隔控制捕获速度
        time.sleep(1.0 / target_fps)

# ========================
# 主处理函数
# ========================
def process_frame(frame):
    global baseline_axes  # 声明使用全局变量
    
    # 转换为RGB格式以供MediaPipe处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 设置图像为只读模式，提高处理效率
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True
    
    if not results.multi_face_landmarks:
        return frame

    # 处理第一个检测到的人脸
    face_landmarks = results.multi_face_landmarks[0]
    
    # 收集平面拟合点位
    pts_plane = cp.array([
        get_landmark_coords(face_landmarks.landmark, idx, frame.shape)
        for idx in PLANE_IDX
    ])
    
    # 绘制平面拟合点
    pts_plane_np = cp.asnumpy(pts_plane)
    for point in pts_plane_np:
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    
    # 拟合平面，得到平面中心和法向量（局部 z 轴）
    centroid, normal = fit_plane(pts_plane)
    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 0), -1)
    
    # 归一化因子：使用所有平面点到质心的平均距离
    distances = cp.linalg.norm(pts_plane - cp.asarray(centroid), axis=1)
    norm_factor = max(float(cp.mean(distances)), 1e-5)  # 确保不为零
    
    # 定义局部坐标系的 X 轴方向：使用索引 117 到 346 两点方向
    vec = pts_plane[1] - pts_plane[0]
    x_axis = vec / max(float(cp.linalg.norm(vec)), 1e-5)  # 避免除零
    
    # 计算 Y 轴：利用右手法则 y = z x x_axis
    y_axis = cp.cross(cp.asarray(normal), x_axis)
    y_axis = y_axis / max(float(cp.linalg.norm(y_axis)), 1e-5)  # 避免除零
    
    # z_axis 已经由拟合的 normal 得到
    z_axis = cp.asarray(normal)
    
    # 如果尚未设置基准坐标系，则初始化
    if baseline_axes is None:
        baseline_axes = (cp.asnumpy(x_axis), cp.asnumpy(y_axis), cp.asnumpy(z_axis))
        print("基准坐标系已初始化")
    
    # 绘制平面和坐标轴
    draw_plane(frame, centroid, cp.asnumpy(x_axis), cp.asnumpy(y_axis), scale=norm_factor, color=(0, 255, 255), alpha=0.3)
    draw_axes(frame, centroid, x_axis, y_axis, z_axis, scale=norm_factor * 2.0)
    
    # 计算欧拉角并显示
    roll, pitch, yaw = compute_euler_angles(x_axis, y_axis, z_axis)
    
    # 显示欧拉角
    cv2.putText(frame, f"Roll: {roll:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# ========================
# 主程序入口
# ========================
def main():
    global fps, frame_times
    
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"D:\Project\数据集\video_\43-MaleNoGlasses-Talking.avi")
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置视频属性 - 固定帧率为30fps
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"设置的目标帧率: {target_fps}, 实际帧率: {actual_fps}")

    # 启动捕获线程
    capture_thread = threading.Thread(target=capture_thread_func, args=(cap, frame_queue, stop_event))
    capture_thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
    capture_thread.start()

    try:
        while True:
            # 记录当前时间用于FPS计算
            start_time = time.time()
            
            if not frame_queue.empty():
                frame = frame_queue.get()
                processed_frame = process_frame(frame)
                cv2.imshow('Face Plane and Euler Angles', processed_frame)
                
                # 更新时间队列用于FPS计算
                frame_times.append(start_time)
                # 计算FPS
                fps = calculate_fps()
            
                # 计算需要等待的时间以维持固定帧率
                processing_time = time.time() - start_time
                wait_time = max(1, int(frame_delay - processing_time * 1000))  # 确保至少等待1ms
                
                # 按 Esc 键退出
                if cv2.waitKey(wait_time) & 0xFF == 27:
                    break
            else:
                # 队列为空时，短暂等待并检查退出条件
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        # 清理资源
        stop_event.set()
        capture_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
