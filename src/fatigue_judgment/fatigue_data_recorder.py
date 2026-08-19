import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime

import matplotlib.animation as animation
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 导入疲劳分析器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fatigue_judgment.fatigue_analyzer import FatigueAnalyzer, FatigueLevel, get_fatigue_level_description

class FatigueDataRecorder:
    """Fatigue data recorder for recording and visualizing eye state, yawn state and head pose data"""
    
    def __init__(self, max_points=300):
        """
        Initialize the data recorder
        
        Parameters:
            max_points: Maximum number of data points, older points will be discarded when exceeded
        """
        self.max_points = max_points
        self.is_running = False
        
        # 线程锁：用于保护共享数据访问
        self.lock = threading.Lock()
        
        # 线程间通信队列
        self.raw_data_queue = queue.Queue()        # 原始数据队列
        self.processed_data_queue = queue.Queue()  # 处理后数据队列
        self.display_data_queue = queue.Queue()    # 显示数据队列
        
        # 线程控制标志
        self.threads_running = False
        
        # Initialize data structures
        self.timestamps = []
        self.eye_states = []  # 0: Eyes open, 1: One eye closed, 2: Both eyes closed
        self.yawn_states = []  # 0: No yawn, 1: Yawning
        self.roll_angles = []
        self.pitch_angles = []
        self.yaw_angles = []
        
        # 初始化所有计数器和状态变量
        self._init_state_variables()
        
        # 疲劳分析器
        self.fatigue_analyzer = FatigueAnalyzer(window_size_minutes=5)
        
        # 线程对象
        self.processing_thread = None
        self.csv_thread = None
        self.gui_thread = None
        self.csv_file = None
        self.csv_writer = None
    
    def _init_state_variables(self):
        """初始化所有状态变量"""
        # 眨眼计数器相关变量
        self.blink_count = 0
        self.last_eye_state = None
        self.last_blink_time = 0
        self.blink_cooldown = 0.6  # 眨眼后的冷却时间（秒）
        self.eye_state_history = []  # 存储最近的眼睛状态历史
        self.eye_state_history_max_len = 4  # 存储4帧的历史记录
        
        # 持续闭眼状态器相关变量
        self.eyes_closed_start_time = None
        self.sustained_eyes_closed = False
        self.sustained_eyes_closed_threshold = 2.0  # 持续闭眼阈值（秒）
        
        # 哈欠计数器相关变量
        self.yawn_count = 0
        self.yawn_start_time = None
        self.yawn_counted = False
        self.yawn_threshold = 2.0  # 持续哈欠阈值（秒）
        
        # 点头计数器相关变量
        self.nod_count = 0
        self.pitch_buffer = []  # 存储所有历史pitch值，用于计算全局平均值
        self.pitch_mean = 0
        self.pitch_std = 0
        self.roll_stable_threshold = 4.0  # 判断pitch稳定的标准差阈值
        self.nod_state = "stable"  # stable, rising, falling
        self.last_nod_time = 0
        self.nod_cooldown = 1.0  # 点头后的冷却时间（秒）
        
        # 低头状态器相关变量
        self.head_down_start_time = None
        self.head_down = False
        self.head_down_threshold = 2.0  # 持续低头阈值（秒）
        self.head_down_angle = 15.0  # 低头角度阈值（相对于平均值）
        
        # 疲劳级别
        self.fatigue_level = FatigueLevel.NORMAL
        self.fatigue_score = 0
        self.blink_rate_per_minute = 0.0
        self.yawn_rate_per_hour = 0.0
        
    def start(self):
        """Start data recording and visualization"""
        if self.is_running:
            return
        
        self.is_running = True
        self.threads_running = True
        
        # Create CSV file
        folder_path = "D:/Project/guaduation_project/data/fatigue_data"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = open(f"{folder_path}/fatigue_data_{timestamp}.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Blink Count', 'Blink Rate(per minute)', 'Yawn Count', 'Yawn Rate(per hour)', 
                                 'Nod Status', 'Head Down Status', 'Fatigue Score', 'Fatigue Level'])
        
        # 重置疲劳分析器
        self.fatigue_analyzer.reset()
        
        # 启动线程
        self._start_threads()
        
        print("All threads started successfully")
    
    def _start_threads(self):
        """启动所有工作线程"""
        # 启动数据处理线程
        self.processing_thread = threading.Thread(target=self._process_data_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 启动CSV记录线程
        self.csv_thread = threading.Thread(target=self._csv_record_thread)
        self.csv_thread.daemon = True
        self.csv_thread.start()
        
        # 启动GUI线程
        self.gui_thread = threading.Thread(target=self._run_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()
    
    def stop(self):
        """Stop data recording and visualization"""
        self.is_running = False
        self.threads_running = False
        
        # 等待所有线程结束
        try:
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            if self.csv_thread and self.csv_thread.is_alive():
                self.csv_thread.join(timeout=1.0)
            # GUI线程通常会在窗口关闭时自动结束
        except Exception as e:
            print(f"Error stopping threads: {e}")
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
    
    def add_data_point(self, eye_state, yawn_state, roll, pitch, yaw):
        """
        Add a data point
        
        Parameters:
            eye_state: "EYES OPEN" / "ONE EYE CLOSED" / "BOTH EYES CLOSED"
            yawn_state: "NO YAWN" / "YAWNING DETECTED"
            roll, pitch, yaw: Euler angles, float values
        """
        if not self.is_running:
            return
        
        # Convert eye state to numeric value
        eye_code = 0  # Default - Eyes open
        if eye_state == "ONE EYE CLOSED":
            eye_code = 1
        elif eye_state == "BOTH EYES CLOSED":
            eye_code = 2
        
        # Convert yawn state to numeric value
        yawn_code = 1 if yawn_state == "YAWNING DETECTED" else 0
        
        # 当前时间
        current_time = time.time()
        
        # 添加原始数据到队列，供处理线程使用
        self.raw_data_queue.put((current_time, eye_code, yawn_code, roll, pitch, yaw))
    
    def _process_data_thread(self):
        """数据处理线程函数"""
        while self.threads_running:
            try:
                # 非阻塞方式获取数据，超时后检查线程状态
                try:
                    current_time, eye_code, yawn_code, roll, pitch, yaw = self.raw_data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 处理各种检测
                self._process_blink_detection(eye_code, current_time)
                self._process_sustained_eyes_closed(eye_code, current_time)
                self._process_yawn_detection(yawn_code, current_time)
                self._update_pitch_statistics(pitch, current_time)
                
                # 更新疲劳分析器
                self._update_fatigue_analyzer()
                
                # 创建处理后的数据点
                processed_data = (
                    current_time, eye_code, yawn_code, roll, pitch, yaw,
                    self.blink_count, int(self.sustained_eyes_closed),
                    self.nod_count, int(self.head_down), self.yawn_count,
                    self.fatigue_level.value, self.fatigue_score
                )
                
                # 将处理后的数据添加到CSV记录队列和显示队列
                self.processed_data_queue.put(processed_data)
                self.display_data_queue.put(processed_data)
                
                # 释放队列任务
                self.raw_data_queue.task_done()
                
            except Exception as e:
                print(f"Error in data processing thread: {e}")
                time.sleep(0.01)  # 避免CPU占用过高
    
    def _process_blink_detection(self, eye_code, current_time):
        """处理眨眼检测逻辑"""
        blink_detected = False
        
        # 添加当前眼睛状态到历史记录
        self.eye_state_history.append(eye_code)
        
        # 保持历史记录在指定长度以内
        if len(self.eye_state_history) > self.eye_state_history_max_len:
            self.eye_state_history.pop(0)
        
        # 在历史窗口中检测眨眼模式（从闭眼到睁眼）
        if len(self.eye_state_history) >= 2:  # 至少需要2个状态才能检测转换
            has_closed_eyes = False
            has_open_eyes = False
            
            # 检查历史中是否有闭眼状态(2)
            for state in self.eye_state_history[:-1]:  # 除了最新状态外的所有历史状态
                if state == 2:
                    has_closed_eyes = True
                    break
            
            # 检查最新状态是否为睁眼(0)
            has_open_eyes = self.eye_state_history[-1] == 0
            
            # 如果检测到从闭眼到睁眼的转换，且满足冷却时间
            if has_closed_eyes and has_open_eyes and (current_time - self.last_blink_time) >= self.blink_cooldown:
                with self.lock:
                    self.blink_count += 1
                self.last_blink_time = current_time
                print(f"Blink detected (window)! Total blinks: {self.blink_count}")
                # 清空历史，避免重复计数
                self.eye_state_history = [eye_code]
                blink_detected = True
        
        # 更新上一次眼睛状态
        self.last_eye_state = eye_code
        
        return blink_detected
    
    def _process_sustained_eyes_closed(self, eye_code, current_time):
        """处理持续闭眼检测逻辑"""
        if eye_code == 2:  # 双眼闭合
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
            elif (current_time - self.eyes_closed_start_time) >= self.sustained_eyes_closed_threshold:
                if not self.sustained_eyes_closed:
                    with self.lock:
                        self.sustained_eyes_closed = True
                    print("Sustained eyes closed detected!")
        else:
            self.eyes_closed_start_time = None
            with self.lock:
                self.sustained_eyes_closed = False
    
    def _process_yawn_detection(self, yawn_code, current_time):
        """处理哈欠检测逻辑"""
        if yawn_code == 1:  # 检测到哈欠
            if self.yawn_start_time is None:
                # 哈欠开始
                self.yawn_start_time = current_time
            elif not self.yawn_counted and (current_time - self.yawn_start_time) >= self.yawn_threshold:
                # 哈欠持续超过阈值且未计数
                with self.lock:
                    self.yawn_count += 1
                    self.yawn_counted = True
                print(f"Yawn detected! Total yawns: {self.yawn_count}")
        else:
            # 哈欠状态结束，重置状态
            self.yawn_start_time = None
            self.yawn_counted = False
    
    def _update_pitch_statistics(self, pitch, current_time):
        """更新pitch角度统计和点头/低头检测"""
        # 添加pitch值到历史记录 - 存储所有检测开始以来的pitch值
        self.pitch_buffer.append(pitch)  # 使用现有的buffer，但存储pitch值
        
        # 至少需要10个样本才能计算稳定的统计值
        if len(self.pitch_buffer) >= 10:
            # 计算全局均值和标准差 - 基于所有历史数据
            with self.lock:
                self.pitch_mean = np.mean(self.pitch_buffer)  # 全局平均值，用于点头和低头检测
                self.pitch_std = np.std(self.pitch_buffer)
            
            # 整合的点头和低头检测逻辑
            if pitch > self.pitch_mean + 15.0:  # 当pitch超过平均值+10度
                if self.head_down_start_time is None:
                    # 首次超过阈值，记录开始时间
                    self.head_down_start_time = current_time
                    self.nod_state = "rising"  # 设置为上升状态
                    print(f"Head motion detected: Rising phase, pitch={pitch:.2f}, mean={self.pitch_mean:.2f}")
                elif not self.head_down:  # 持续超过阈值但尚未设为低头状态
                    # 检查持续时间
                    duration = current_time - self.head_down_start_time
                    if duration >= self.head_down_threshold:  # 持续超过2秒
                        # 判定为低头
                        with self.lock:
                            self.head_down = True
                        print(f"Head down detected! Duration: {duration:.2f}s")
            else:  # pitch回到阈值以下
                if self.head_down_start_time is not None:
                    # 之前处于超过阈值状态，现在回落
                    duration = current_time - self.head_down_start_time
                    
                    if self.head_down:
                        # 如果之前判定为低头状态，现在重置
                        with self.lock:
                            self.head_down = False
                        print(f"Head up detected after {duration:.2f}s")
                    elif duration < self.head_down_threshold and self.nod_state == "rising":
                        # 短暂超过阈值后回落，判定为点头
                        if (current_time - self.last_nod_time) >= self.nod_cooldown:
                            with self.lock:
                                self.nod_count += 1
                            self.last_nod_time = current_time
                            print(f"Nod detected! Duration: {duration:.2f}s, Total nods: {self.nod_count}")
                    
                    # 重置状态
                    self.head_down_start_time = None
                    self.nod_state = "stable"
    
    # 向后兼容的方法
    def _update_roll_statistics(self, roll, current_time):
        """更新roll角度统计和点头/低头检测（已弃用，保留为向后兼容）"""
        # 将调用重定向到新方法，但使用pitch参数
        print("警告：_update_roll_statistics方法已弃用，请使用_update_pitch_statistics")
        # 此处不做任何操作，因为_process_data_thread现在直接调用_update_pitch_statistics
        pass
    
    def _update_fatigue_analyzer(self):
        """更新疲劳分析器并获取最新的疲劳等级和评分"""
        metrics = {
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "nod_count": self.nod_count,
            "head_down": self.head_down,
            "sustained_eyes_closed": self.sustained_eyes_closed
        }
        
        # 更新分析器
        self.fatigue_level = self.fatigue_analyzer.update(metrics)
        
        # 获取详细统计并使用分析器计算的频率
        stats = self.fatigue_analyzer.get_fatigue_stats()
        self.fatigue_score = stats["fatigue_score"]
        self.blink_rate_per_minute = stats["blink_rate_per_minute"]
        self.yawn_rate_per_hour = stats["yawn_rate_per_hour"]
    
    def _update_data(self):
        """从显示队列更新数据到可视化组件"""
        # 处理所有待显示数据
        display_data_list = []
        while not self.display_data_queue.empty():
            try:
                display_data_list.append(self.display_data_queue.get_nowait())
                self.display_data_queue.task_done()
            except queue.Empty:
                break
        
        # 如果有数据需要处理
        if display_data_list:
            for data in display_data_list:
                # 解包数据
                timestamp, eye_code, yawn_code, roll, pitch, yaw = data[:6]
                
                # 更新显示数据列表
                with self.lock:
                    self.timestamps.append(timestamp)
                    self.eye_states.append(eye_code)
                    self.yawn_states.append(yawn_code)
                    self.roll_angles.append(roll)
                    self.pitch_angles.append(pitch)
                    self.yaw_angles.append(yaw)
                    
                    # 限制数据点数量
                    if len(self.timestamps) > self.max_points:
                        self.timestamps = self.timestamps[-self.max_points:]
                        self.eye_states = self.eye_states[-self.max_points:]
                        self.yawn_states = self.yawn_states[-self.max_points:]
                        self.roll_angles = self.roll_angles[-self.max_points:]
                        self.pitch_angles = self.pitch_angles[-self.max_points:]
                        self.yaw_angles = self.yaw_angles[-self.max_points:]
    
    def _run_gui(self):
        """GUI线程函数"""
        # Create Tkinter window
        root = tk.Tk()
        root.title("Fatigue Data Visualization")
        root.geometry("1000x800")
        root.protocol("WM_DELETE_WINDOW", lambda: self._on_closing(root))
        
        # 初始化频率指标
        self.blink_rate_per_minute = 0.0
        self.yawn_rate_per_hour = 0.0
        
        # Create Figure
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Layout subplots
        ax1 = fig.add_subplot(3, 1, 1)  # Eye and yawn states
        ax2 = fig.add_subplot(3, 1, 2)  # Euler angles - Roll & Pitch
        ax3 = fig.add_subplot(3, 1, 3)  # Euler angles - Yaw
        
        # Initialize empty plots
        eye_line, = ax1.plot([], [], 'b-', label='Eye State')
        yawn_line, = ax1.plot([], [], 'r-', label='Yawn State')
        roll_line, = ax2.plot([], [], 'g-', label='Roll')
        pitch_line, = ax2.plot([], [], 'm-', label='Pitch')
        yaw_line, = ax3.plot([], [], 'c-', label='Yaw')
        
        # Set chart titles and labels
        ax1.set_title('Eye and Yawn States')
        ax1.set_ylabel('State')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Eyes Open', 'One Eye Closed', 'Both Eyes Closed'])
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Head Pose - Roll & Pitch')
        ax2.set_ylabel('Angle (°)')
        ax2.legend()
        ax2.grid(True)
        
        ax3.set_title('Head Pose - Yaw')
        ax3.set_xlabel('Time (seconds ago)')
        ax3.set_ylabel('Angle (°)')
        ax3.legend()
        ax3.grid(True)
        
        # 预设X轴范围和刻度以确保初始显示正确
        x_min = -10.0
        x_max = 0.5
        x_ticks = [-10, -8, -6, -4, -2, 0]
        x_tick_labels = ['10s', '8s', '6s', '4s', '2s', 'now']
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
        
        # 设置更合适的Y轴范围
        ax1.set_ylim(-0.5, 2.5)
        ax2.set_ylim(-30, 30)  # 头部姿态角度通常不会达到90度，使用更小的范围
        ax3.set_ylim(-30, 30)
        
        # 设置网格线更清晰
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 创建状态显示区域
        status_frame = tk.Frame(root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # 状态标签
        blink_label = tk.Label(status_frame, text="Blinks: 0", font=("Arial", 12))
        blink_label.grid(row=0, column=0, padx=10)
        
        eyes_closed_label = tk.Label(status_frame, text="Sustained Eyes Closed: No", font=("Arial", 12))
        eyes_closed_label.grid(row=0, column=1, padx=10)
        
        # 添加哈欠计数显示
        yawn_label = tk.Label(status_frame, text="Yawns: 0", font=("Arial", 12))
        yawn_label.grid(row=0, column=2, padx=10)
        
        nod_label = tk.Label(status_frame, text="Nods: 0", font=("Arial", 12))
        nod_label.grid(row=1, column=0, padx=10)
        
        head_down_label = tk.Label(status_frame, text="Head Down: No", font=("Arial", 12))
        head_down_label.grid(row=1, column=1, padx=10)
        
        # 添加pitch平均值显示
        pitch_mean_label = tk.Label(status_frame, text="Pitch Mean: 0.00°", font=("Arial", 12))
        pitch_mean_label.grid(row=1, column=2, padx=10)
        
        # 添加疲劳评分显示
        fatigue_frame = tk.Frame(root, bg="#222222")
        fatigue_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # 疲劳等级标签
        fatigue_level_label = tk.Label(fatigue_frame, text="疲劳等级: 正常", 
                                     font=("Arial", 16, "bold"), bg="#222222", fg="#00FF00")
        fatigue_level_label.pack(side=tk.LEFT, padx=20)
        
        # 疲劳评分标签
        fatigue_score_label = tk.Label(fatigue_frame, text="疲劳评分: 0/100", 
                                      font=("Arial", 16, "bold"), bg="#222222", fg="#00FF00")
        fatigue_score_label.pack(side=tk.RIGHT, padx=20)
        
        # 添加更详细的疲劳指标显示
        stats_frame = tk.Frame(root)
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        blink_rate_label = tk.Label(stats_frame, text="眨眼频率: 0.0 次/分钟", font=("Arial", 10))
        blink_rate_label.grid(row=0, column=0, padx=10)
        
        yawn_rate_label = tk.Label(stats_frame, text="哈欠频率: 0.0 次/小时", font=("Arial", 10))
        yawn_rate_label.grid(row=0, column=1, padx=10)
        
        # 立即绘制以确保初始设置生效
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Animation function
        def animate(i):
            # 更新数据
            self._update_data()
            
            with self.lock:
                if not self.timestamps:
                    return eye_line, yawn_line, roll_line, pitch_line, yaw_line
                
                # 简化时间计算 - 直接使用相对于最新时间的秒数
                current_time = max(self.timestamps)
                rel_times = [-(current_time - t) for t in self.timestamps]
                
                # 更新图表数据 - 直接使用所有数据点
                eye_line.set_data(rel_times, self.eye_states)
                yawn_line.set_data(rel_times, self.yawn_states)
                roll_line.set_data(rel_times, self.roll_angles)
                pitch_line.set_data(rel_times, self.pitch_angles)
                yaw_line.set_data(rel_times, self.yaw_angles)
                
                # 更新状态标签
                blink_count = self.blink_count
                sustained_eyes_closed = self.sustained_eyes_closed
                nod_count = self.nod_count
                head_down = self.head_down
                pitch_mean = self.pitch_mean
                yawn_count = self.yawn_count
                fatigue_level = self.fatigue_level
                fatigue_score = self.fatigue_score
                
                # 获取更详细的疲劳统计数据
                blink_rate = self.blink_rate_per_minute
                yawn_rate = self.yawn_rate_per_hour
            
            # 固定X轴范围
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(-10.0, 0.5)
            
            # 固定Y轴范围
            ax1.set_ylim(-0.5, 2.5)
            ax2.set_ylim(-30, 30)
            ax3.set_ylim(-30, 30)
            
            # 更新状态标签（在锁之外更新UI）
            blink_label.config(text=f"Blinks: {blink_count}")
            eyes_closed_label.config(text=f"Sustained Eyes Closed: {'Yes' if sustained_eyes_closed else 'No'}")
            yawn_label.config(text=f"Yawns: {yawn_count}")
            nod_label.config(text=f"Nods: {nod_count}")
            head_down_label.config(text=f"Head Down: {'Yes' if head_down else 'No'}")
            pitch_mean_label.config(text=f"Pitch Mean: {pitch_mean:.2f}°")
            
            # 更新疲劳指标显示
            fatigue_level_text = get_fatigue_level_description(fatigue_level)
            fatigue_level_label.config(text=f"疲劳等级: {fatigue_level_text}")
            fatigue_score_label.config(text=f"疲劳评分: {fatigue_score:.1f}/100")
            
            # 根据疲劳等级更新颜色
            if fatigue_level == FatigueLevel.NORMAL:
                color = "#00FF00"  # 绿色
            elif fatigue_level == FatigueLevel.MILD:
                color = "#FFFF00"  # 黄色
            elif fatigue_level == FatigueLevel.MODERATE:
                color = "#FFA500"  # 橙色
            elif fatigue_level == FatigueLevel.SEVERE:
                color = "#FF0000"  # 红色
            else:  # 危险驾驶
                color = "#FF00FF"  # 紫色
            
            fatigue_level_label.config(fg=color)
            fatigue_score_label.config(fg=color)
            
            # 更新详细统计信息
            blink_rate_label.config(text=f"眨眼频率: {blink_rate:.1f} 次/分钟")
            yawn_rate_label.config(text=f"哈欠频率: {yawn_rate:.1f} 次/小时")
            
            # 禁用blitting以确保完整重绘
            return eye_line, yawn_line, roll_line, pitch_line, yaw_line
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, animate, interval=100, blit=False)
        
        root.mainloop()
    
    def _on_closing(self, root):
        """Window closing handler"""
        self.stop()
        root.destroy()
    
    def get_fatigue_metrics(self):
        """获取疲劳驾驶相关指标"""
        with self.lock:
            return {
                "blink_count": self.blink_count,
                "sustained_eyes_closed": self.sustained_eyes_closed,
                "nod_count": self.nod_count,
                "head_down": self.head_down,
                "pitch_mean": self.pitch_mean if hasattr(self, 'pitch_mean') else 0,
                "pitch_std": self.pitch_std if hasattr(self, 'pitch_std') else 0,
                "yawn_count": self.yawn_count,
                "fatigue_level": self.fatigue_level,
                "fatigue_score": self.fatigue_score,
                "blink_rate_per_minute": self.blink_rate_per_minute,
                "yawn_rate_per_hour": self.yawn_rate_per_hour
            }
    
    def _csv_record_thread(self):
        """CSV数据记录线程函数"""
        # 添加缓存字典，按秒分组存储数据
        second_cache = {}
        last_flush_time = time.time()
        
        while self.threads_running:
            try:
                # 非阻塞方式获取数据，超时后检查线程状态
                try:
                    data = self.processed_data_queue.get(timeout=0.1)
                except queue.Empty:
                    # 当队列为空且距离上次写入已经过去1秒，则写入缓存数据
                    current_time = time.time()
                    if current_time - last_flush_time >= 1.0 and second_cache:
                        self._flush_cache_to_csv(second_cache)
                        second_cache.clear()
                        last_flush_time = current_time
                    continue
                
                # 解包数据
                current_time, eye_code, yawn_code, roll, pitch, yaw, blink_count, sustained_eyes_closed, nod_count, head_down, yawn_count, fatigue_level, fatigue_score = data
                
                # 将时间精度转换为秒级别（去掉小数部分）
                second_key = int(current_time)
                
                # 缓存当前秒的数据（会覆盖同一秒内的之前帧）
                second_cache[second_key] = data
                
                # 如果距离上次写入已经过去1秒以上，则写入缓存并清空
                if current_time - last_flush_time >= 1.0:
                    self._flush_cache_to_csv(second_cache)
                    second_cache.clear()
                    last_flush_time = current_time
                
                # 释放队列任务
                self.processed_data_queue.task_done()
                
            except Exception as e:
                print(f"Error in CSV recording thread: {e}")
                time.sleep(0.01)  # 避免CPU占用过高
    
    def _flush_cache_to_csv(self, second_cache):
        """将缓存的数据写入CSV文件（只写入每秒最后一帧）"""
        if not second_cache or not self.csv_file or not self.csv_writer:
            return
        
        # 按时间戳排序
        sorted_timestamps = sorted(second_cache.keys())
        
        for timestamp_sec in sorted_timestamps:
            # 获取该秒的数据
            data = second_cache[timestamp_sec]
            current_time, eye_code, yawn_code, roll, pitch, yaw, blink_count, sustained_eyes_closed, nod_count, head_down, yawn_count, fatigue_level, fatigue_score = data
            
            # 将时间戳转换为秒级别的字符串格式
            timestamp_str = datetime.fromtimestamp(timestamp_sec).strftime("%Y-%m-%d %H:%M:%S")
            
            # 获取疲劳等级的描述文本
            fatigue_level_text = get_fatigue_level_description(self.fatigue_level)
            
            # 点头状态和低头状态转换为文本
            nod_status = "1" if nod_count > 0 else "0"
            head_down_status = "1" if head_down else "0"
            
            # 写入新的CSV格式
            self.csv_writer.writerow([
                timestamp_str,
                blink_count, 
                round(self.blink_rate_per_minute, 2),
                yawn_count, 
                round(self.yawn_rate_per_hour, 2),
                nod_status,
                head_down_status,
                round(fatigue_score, 1),
                fatigue_level_text
            ])
        
        # 及时将数据刷新到文件
        self.csv_file.flush()


# Example usage
if __name__ == "__main__":
    # Create recorder
    recorder = FatigueDataRecorder()
    recorder.start()
    
    # Simulate data generation
    try:
        for i in range(1000):
            # Simulate eye state changes
            if i % 100 < 70:
                eye_state = "EYES OPEN"
            elif i % 100 < 85:
                eye_state = "ONE EYE CLOSED"
            else:
                eye_state = "BOTH EYES CLOSED"
            
            # Simulate yawn state changes
            yawn_state = "YAWNING DETECTED" if i % 150 > 120 else "NO YAWN"
            
            # Simulate Euler angle changes
            roll = 10 * np.sin(i / 20)
            pitch = 15 * np.cos(i / 30)
            yaw = 20 * np.sin(i / 50)
            
            recorder.add_data_point(eye_state, yawn_state, roll, pitch, yaw)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    recorder.stop() 