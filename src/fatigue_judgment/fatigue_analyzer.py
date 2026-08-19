import threading
import time
from enum import Enum


class FatigueLevel(Enum):
    """疲劳等级枚举"""
    NORMAL = 0       # 正常
    MILD = 1         # 轻度疲劳
    MODERATE = 2     # 中度疲劳
    SEVERE = 3       # 重度疲劳
    DANGEROUS = 4    # 危险驾驶


class FatigueAnalyzer:
    """疲劳分析器，用于评估驾驶员的疲劳程度"""
    
    def __init__(self, window_size_minutes=5):
        """
        初始化疲劳分析器
        
        参数:
            window_size_minutes: 用于计算频率的时间窗口大小（分钟）
        """
        self.window_size_minutes = window_size_minutes
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 滑动窗口相关变量
        self.initial_blink_count = 0
        self.initial_yawn_count = 0
        self.initial_nod_count = 0
        self.window_start_time = time.time()
        self.last_reset_time = self.window_start_time
        
        # 状态变量
        self.head_down = False
        self.eyes_closed = False
        self.current_fatigue_level = FatigueLevel.NORMAL
        self.current_fatigue_score = 0
        self.current_blink_rate = 0
        self.current_yawn_rate = 0
        
        # 阈值设置
        # 眨眼频率阈值（次/分钟）
        self.blink_thresholds = {
            "normal": (10, 15),      # 正常：10-15次/分钟
            "mild": (15, 25),        # 轻度疲劳：15-25次/分钟
            "moderate": (25, 30),    # 中度疲劳：25-30次/分钟
            "severe": 30             # 重度疲劳：>35次/分钟
        }
        
        # 哈欠频率阈值（次/小时）
        self.yawn_thresholds = {
            "normal": (0.2, 0.4),    # 正常：0.2-0.4次/小时
            "mild": (1, 2),          # 轻度疲劳：1-2次/小时
            "moderate": (3, 5),      # 中度疲劳：3-5次/小时
            "severe": 5              # 重度疲劳：>5次/小时
        }
        
    def update(self, metrics):
        """
        更新疲劳分析器状态并计算当前疲劳等级
        
        参数:
            metrics: 包含当前检测到的各项指标的字典
                {
                    "blink_count": 眨眼计数,
                    "yawn_count": 哈欠计数,
                    "nod_count": 点头计数,
                    "head_down": 低头状态(布尔值),
                    "sustained_eyes_closed": 持续闭眼状态(布尔值)
                }
        
        返回:
            当前疲劳等级（FatigueLevel枚举）
        """
        with self.lock:
            # 更新状态
            self.head_down = metrics["head_down"]
            self.eyes_closed = metrics["sustained_eyes_closed"]
            
            # 检查是否需要重置计数基准（如果程序运行超过1小时）
            current_time = time.time()
            window_size_seconds = 3600  # 滑动窗口大小为1小时
            
            if current_time - self.last_reset_time > window_size_seconds:
                # 更新初始计数为当前计数，这样在计算频率时只考虑窗口内的增量
                self.initial_blink_count = metrics["blink_count"]
                self.initial_yawn_count = metrics["yawn_count"]
                self.initial_nod_count = metrics["nod_count"]
                
                # 更新窗口起始时间和最后重置时间
                self.window_start_time = current_time
                self.last_reset_time = current_time
                
                print(f"Sliding window updated - Time: {time.strftime('%H:%M:%S', time.localtime(current_time))}")
                print(f"Initial blink count: {self.initial_blink_count}, Initial yawn count: {self.initial_yawn_count}")
            
            # 计算疲劳等级
            recent_nod = metrics["nod_count"] > self.initial_nod_count
            self._calculate_fatigue_level(
                metrics["blink_count"],
                metrics["yawn_count"],
                metrics["sustained_eyes_closed"], 
                metrics["head_down"],
                recent_nod
            )
            
            return self.current_fatigue_level
    
    def _calculate_fatigue_level(self, blink_count, yawn_count, eyes_closed, head_down, recent_nod):
        """
        根据各项指标计算疲劳等级
        
        参数:
            blink_count: 眨眼总计数
            yawn_count: 哈欠总计数
            eyes_closed: 持续闭眼状态
            head_down: 低头状态
            recent_nod: 最近是否有点头行为
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
        
        # 特殊情况判断：重度疲劳
        if eyes_closed and recent_nod:
            self.current_fatigue_level = FatigueLevel.SEVERE
            self.current_fatigue_score = 90
            return
        
        # 计算眨眼和哈欠的疲劳评分
        blink_score = self._calculate_score(
            self.current_blink_rate, 
            self.blink_thresholds, 
            self.blink_thresholds["severe"] + 5
        )
        
        yawn_score = self._calculate_score(
            self.current_yawn_rate, 
            self.yawn_thresholds, 
            self.yawn_thresholds["severe"] + 2
        )
        
        # 组合评分，赋予眨眼和哈欠不同的权重
        # 眨眼权重: 0.6, 哈欠权重: 0.4
        combined_score = blink_score * 0.6 + yawn_score * 0.4
        
        # 根据持续闭眼和低头状态调整分数
        if eyes_closed:
            combined_score += 15
        
        if head_down:
            combined_score += 15
        
        # 限制最大分数为100
        combined_score = min(100, combined_score)
        
        # 根据组合分数确定最终疲劳等级
        if combined_score >= 85:
            self.current_fatigue_level = FatigueLevel.SEVERE
        elif combined_score >= 65:
            self.current_fatigue_level = FatigueLevel.MODERATE
        elif combined_score >= 40:
            self.current_fatigue_level = FatigueLevel.MILD
        else:
            self.current_fatigue_level = FatigueLevel.NORMAL
        
        self.current_fatigue_score = combined_score
    
    def _calculate_score(self, value, thresholds, max_value):
        """
        根据给定的值和阈值计算评分
        
        参数:
            value: 要评分的值
            thresholds: 阈值字典，包含"normal", "mild", "moderate", "severe"键
            max_value: 考虑的最大值
            
        返回:
            评分（0-100）
        """
        if value <= thresholds["normal"][0]:
            # 低于正常下限，最低分数10
            return 10
        elif value <= thresholds["normal"][1]:
            # 正常范围内，线性从10到30
            ratio = (value - thresholds["normal"][0]) / (thresholds["normal"][1] - thresholds["normal"][0])
            return 10 + ratio * 20
        elif value <= thresholds["mild"][1]:
            # 轻度疲劳范围，线性从30到60
            ratio = (value - thresholds["normal"][1]) / (thresholds["mild"][1] - thresholds["normal"][1])
            return 30 + ratio * 30
        elif value <= thresholds["moderate"][1]:
            # 中度疲劳范围，线性从60到80
            ratio = (value - thresholds["mild"][1]) / (thresholds["moderate"][1] - thresholds["mild"][1])
            return 60 + ratio * 20
        else:
            # 超过中度疲劳上限，线性从80到90
            if value >= max_value:
                return 90
            else:
                ratio = (value - thresholds["moderate"][1]) / (max_value - thresholds["moderate"][1])
                return 80 + ratio * 10
    
    def get_fatigue_stats(self):
        """
        获取当前疲劳统计数据
        
        返回:
            包含各项统计指标的字典
        """
        with self.lock:
            current_time = time.time()
            elapsed_minutes = max(1.0, (current_time - self.window_start_time) / 60)
            elapsed_hours = max(1.0/60, elapsed_minutes / 60)
            
            return {
                "fatigue_level": self.current_fatigue_level,
                "fatigue_score": self.current_fatigue_score,
                "blink_rate_per_minute": self.current_blink_rate,
                "yawn_rate_per_hour": self.current_yawn_rate,
                "eyes_closed": self.eyes_closed,
                "head_down": self.head_down,
                "elapsed_minutes": elapsed_minutes,
                "window_start_time": self.window_start_time
            }
    
    def reset(self):
        """重置分析器状态"""
        with self.lock:
            self.initial_blink_count = 0
            self.initial_yawn_count = 0
            self.initial_nod_count = 0
            self.head_down = False
            self.eyes_closed = False
            self.current_fatigue_level = FatigueLevel.NORMAL
            self.current_fatigue_score = 0
            self.window_start_time = time.time()
            self.last_reset_time = self.window_start_time
            self.current_blink_rate = 0
            self.current_yawn_rate = 0


def get_fatigue_level_description(fatigue_level):
    """
    Get the text description of fatigue level
    
    Parameters:
        fatigue_level: FatigueLevel enum value
    
    Returns:
        Text description of fatigue level
    """
    descriptions = {
        FatigueLevel.NORMAL: "Normal",
        FatigueLevel.MILD: "Mild Fatigue",
        FatigueLevel.MODERATE: "Moderate Fatigue",
        FatigueLevel.SEVERE: "Severe Fatigue",
        FatigueLevel.DANGEROUS: "Dangerous Driving"
    }
    return descriptions.get(fatigue_level, "Unknown")


# 示例用法
if __name__ == "__main__":
    # 创建疲劳分析器，使用5分钟的时间窗口
    analyzer = FatigueAnalyzer(window_size_minutes=5)
    
    # 模拟更新指标
    metrics = {
        "blink_count": 30,  # 总眨眼次数
        "yawn_count": 2,    # 总哈欠次数
        "nod_count": 1,     # 总点头次数
        "head_down": False, # 是否低头
        "sustained_eyes_closed": False  # 是否持续闭眼
    }
    
    # 更新分析器并获取疲劳等级
    fatigue_level = analyzer.update(metrics)
    
    # 获取详细统计信息
    stats = analyzer.get_fatigue_stats()
    
    # 打印结果
    print(f"疲劳等级: {get_fatigue_level_description(fatigue_level)}")
    print(f"疲劳评分: {stats['fatigue_score']:.1f}/100")
    print(f"每分钟眨眼频率: {stats['blink_rate_per_minute']:.1f} 次/分钟")
    print(f"每小时哈欠频率: {stats['yawn_rate_per_hour']:.1f} 次/小时")
