import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def get_fatigue_level_name(level):
    """获取疲劳等级的英文名称"""
    level_names = {
        0: "Normal",
        1: "Mild",
        2: "Moderate", 
        3: "Severe",
        4: "Dangerous"
    }
    return level_names.get(level, "Unknown")


def calculate_fatigue_level(blink_rate, yawn_rate, head_down, eyes_closed):
    """根据生理指标计算疲劳等级，模拟FatigueAnalyzer的计算逻辑"""
    # 特殊情况：如果同时观察到头部下垂和持续闭眼，判定为危险状态
    if head_down and eyes_closed:
        return 4, 100  # Dangerous
    
    # 特殊规则：如果眨眼频率大于30且哈欠频率大于5，直接判定为重度疲劳
    if blink_rate > 30 and yawn_rate > 5:
        return 3, 90  # Severe
    
    # 眨眼频率阈值（次/分钟）
    blink_thresholds = {
        "normal": (10, 15),    # 正常：10-15次/分钟
        "mild": (15, 25),      # 轻度疲劳：15-25次/分钟
        "moderate": (25, 30),  # 中度疲劳：25-30次/分钟
        "severe": 30           # 重度疲劳：>30次/分钟
    }
    
    # 哈欠频率阈值（次/小时）
    yawn_thresholds = {
        "normal": (0.2, 0.4),  # 正常：0.2-0.4次/小时
        "mild": (1, 2),        # 轻度疲劳：1-2次/小时
        "moderate": (3, 5),    # 中度疲劳：3-5次/小时
        "severe": 5            # 重度疲劳：>5次/小时
    }
    
    # 根据眨眼频率计算疲劳分数
    if blink_rate <= blink_thresholds["normal"][0]:
        blink_score = 10
    elif blink_rate <= blink_thresholds["normal"][1]:
        ratio = (blink_rate - blink_thresholds["normal"][0]) / (blink_thresholds["normal"][1] - blink_thresholds["normal"][0])
        blink_score = 10 + ratio * 20
    elif blink_rate <= blink_thresholds["mild"][1]:
        ratio = (blink_rate - blink_thresholds["normal"][1]) / (blink_thresholds["mild"][1] - blink_thresholds["normal"][1])
        blink_score = 30 + ratio * 30
    elif blink_rate <= blink_thresholds["moderate"][1]:
        ratio = (blink_rate - blink_thresholds["mild"][1]) / (blink_thresholds["moderate"][1] - blink_thresholds["mild"][1])
        blink_score = 60 + ratio * 20
    else:
        max_rate = blink_thresholds["severe"] + 5  # 使用30+5=35作为最大考虑值
        if blink_rate >= max_rate:
            blink_score = 90
        else:
            ratio = (blink_rate - blink_thresholds["moderate"][1]) / (max_rate - blink_thresholds["moderate"][1])
            blink_score = 80 + ratio * 10
    
    # 根据哈欠频率计算疲劳分数
    if yawn_rate <= yawn_thresholds["normal"][0]:
        yawn_score = 10
    elif yawn_rate <= yawn_thresholds["normal"][1]:
        ratio = (yawn_rate - yawn_thresholds["normal"][0]) / (yawn_thresholds["normal"][1] - yawn_thresholds["normal"][0])
        yawn_score = 10 + ratio * 20
    elif yawn_rate <= yawn_thresholds["mild"][1]:
        ratio = (yawn_rate - yawn_thresholds["normal"][1]) / (yawn_thresholds["mild"][1] - yawn_thresholds["normal"][1])
        yawn_score = 30 + ratio * 30
    elif yawn_rate <= yawn_thresholds["moderate"][1]:
        ratio = (yawn_rate - yawn_thresholds["mild"][1]) / (yawn_thresholds["moderate"][1] - yawn_thresholds["mild"][1])
        yawn_score = 60 + ratio * 20
    else:
        max_rate = yawn_thresholds["severe"] + 2
        if yawn_rate >= max_rate:
            yawn_score = 90
        else:
            ratio = (yawn_rate - yawn_thresholds["moderate"][1]) / (max_rate - yawn_thresholds["moderate"][1])
            yawn_score = 80 + ratio * 10
    
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
        fatigue_level = 3  # Severe
    elif combined_score >= 65:
        fatigue_level = 2  # Moderate
    elif combined_score >= 40:
        fatigue_level = 1  # Mild
    else:
        fatigue_level = 0  # Normal
    
    # 额外的阈值规则：超过单项阈值也直接判断为对应疲劳等级
    if blink_rate > blink_thresholds["severe"]:
        fatigue_level = max(fatigue_level, 3)  # 至少是重度疲劳
        combined_score = max(combined_score, 85)
    elif blink_rate > blink_thresholds["moderate"][1]:
        fatigue_level = max(fatigue_level, 2)  # 至少是中度疲劳
        combined_score = max(combined_score, 65)
    
    if yawn_rate > yawn_thresholds["severe"]:
        fatigue_level = max(fatigue_level, 3)  # 至少是重度疲劳
        combined_score = max(combined_score, 85)
    elif yawn_rate > yawn_thresholds["moderate"][1]:
        fatigue_level = max(fatigue_level, 2)  # 至少是中度疲劳
        combined_score = max(combined_score, 65)
    
    return fatigue_level, combined_score


def generate_fatigue_data(duration_hours=None):
    """生成连续每秒的虚拟疲劳数据，疲劳增长速度随机，不一定达到重度疲劳状态"""
    # 如果未指定时长，则随机生成4-6小时
    if duration_hours is None:
        duration_hours = random.uniform(4.0, 6.0)

    # 转换为秒数
    total_seconds = int(duration_hours * 3600)
    
    # 每秒记录一次数据
    num_records = total_seconds

    # 创建时间戳列表
    start_time = datetime.now() - timedelta(hours=duration_hours)
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_records)]

    # 随机决定驾驶过程的最终疲劳状态和增长模式
    growth_pattern = random.choice(["slow", "medium", "fast"])
    final_fatigue_level = random.choice([1, 2, 3])  # 1=Mild, 2=Moderate, 3=Severe
    
    # 提高相关系数，确保疲劳状态下更强的相关性
    base_correlation = random.uniform(0.85, 0.95)  # 提高基础相关性至85%-95%
    
    # 相关性随疲劳状态增强的函数
    def get_dynamic_correlation(fatigue_progress):
        # 疲劳程度越高，相关性越强
        return min(0.98, base_correlation + fatigue_progress * 0.15)
    
    # 生成基础疲劳曲线（作为眨眼和哈欠频率的共同基础）
    t = np.linspace(0, 1, num_records)
    
    # 设置不同增长模式的指数参数
    if growth_pattern == "slow":
        growth_exponent = 1.1  # 缓慢增长，接近线性
        fluctuation_scale = 2.0  # 大波动
    elif growth_pattern == "medium":
        growth_exponent = 1.5  # 中等增长速度
        fluctuation_scale = 1.5  # 中等波动
    else:  # fast
        growth_exponent = 2.5  # 快速增长
        fluctuation_scale = 1.0  # 小波动
    
    # 根据最终疲劳等级确定最大眨眼频率和哈欠频率
    if final_fatigue_level == 1:  # Mild
        max_blink_rate = random.uniform(18, 23)  # 轻度疲劳，眨眼率在15-25范围内
        max_yawn_rate = random.uniform(1.5, 2.0)  # 轻度疲劳，哈欠率在1-2范围内
    elif final_fatigue_level == 2:  # Moderate
        max_blink_rate = random.uniform(25, 29)  # 中度疲劳，眨眼率在25-30范围内
        max_yawn_rate = random.uniform(3, 5)     # 中度疲劳，哈欠率在3-5范围内
    else:  # Severe
        max_blink_rate = random.uniform(32, 38)  # 重度疲劳，眨眼率>30
        max_yawn_rate = random.uniform(6, 8)     # 重度疲劳，哈欠率>5
    
    # 生成基础疲劳进展曲线(0-1范围)
    base_fatigue_curve = t**growth_exponent
    
    # 添加随机波动到基础疲劳曲线
    base_noise = np.random.normal(0, 0.05, num_records)
    smoothing_window = np.ones(40) / 40
    base_noise = np.convolve(base_noise, smoothing_window, mode='same')
    base_fatigue_curve += base_noise
    base_fatigue_curve = np.clip(base_fatigue_curve, 0, 1)  # 确保值在0-1范围内
    
    # 基于共享基础曲线生成眨眼频率曲线
    blink_rate_baseline = 13 + (max_blink_rate - 13) * base_fatigue_curve
    
    # 生成独立的随机波动部分（不相关部分）
    blink_independent_noise = np.random.normal(0, fluctuation_scale * (1 - get_dynamic_correlation(base_fatigue_curve[-1])), num_records)
    blink_independent_noise = np.convolve(blink_independent_noise, smoothing_window, mode='same')
    
    # 添加周期性变化
    cycle_amplitude = random.uniform(0.5, 1.0)
    low_freq_cycle = cycle_amplitude * np.sin(2 * np.pi * t * 3) + 0.5 * np.cos(2 * np.pi * t * 5)
    
    # 组合生成最终眨眼频率曲线（基础曲线 + 独立波动 + 周期性）
    blink_rate_curve = blink_rate_baseline + blink_independent_noise + low_freq_cycle
    blink_rate_curve = np.clip(blink_rate_curve, 12.3, max_blink_rate + 2)
    
    # 创建哈欠频率的映射曲线
    # 首先从基础疲劳曲线派生出连续的哈欠频率基线
    yawn_continuous_baseline = 0.1 + (max_yawn_rate - 0.1) * base_fatigue_curve
    
    # 添加独立噪声使哈欠频率与眨眼频率有一定偏差
    yawn_independent_noise = np.random.normal(0, 0.4 * (1 - get_dynamic_correlation(base_fatigue_curve[-1])), num_records)
    yawn_independent_noise = np.convolve(yawn_independent_noise, np.ones(100) / 100, mode='same')  # 使用较大的平滑窗口
    
    # 连续哈欠频率曲线
    yawn_continuous_curve = yawn_continuous_baseline + yawn_independent_noise
    yawn_continuous_curve = np.clip(yawn_continuous_curve, 0.05, max_yawn_rate + 1)
    
    # 将连续哈欠频率转换为离散阶段
    # 定义哈欠频率的离散值
    yawn_discrete_values = [0, 1, 2, 3, 4, 6, 8]
    
    # 创建离散哈欠频率曲线
    yawn_rate_curve = np.zeros(num_records)
    
    # 根据连续曲线的值范围将其映射到最接近的离散值
    for i in range(num_records):
        # 找到最接近的离散值
        closest_value = min(yawn_discrete_values, key=lambda x: abs(x - yawn_continuous_curve[i]))
        yawn_rate_curve[i] = closest_value
    
    # 平滑哈欠频率曲线，避免频繁跳跃
    transition_length = 60  # 60秒的过渡期
    for i in range(1, num_records):
        if yawn_rate_curve[i] != yawn_rate_curve[i-1]:
            # 只有在有足够空间进行平滑时才进行
            if i + transition_length < num_records:
                # 找到下一个不同的值
                next_diff_idx = i
                while next_diff_idx < num_records and yawn_rate_curve[next_diff_idx] == yawn_rate_curve[i]:
                    next_diff_idx += 1
                
                # 如果找到了且有足够空间，则创建平滑过渡
                if next_diff_idx < num_records and next_diff_idx - i > transition_length:
                    current_value = yawn_rate_curve[i-1]
                    next_value = yawn_rate_curve[i]
                    # 在转换期内保持前一个值
                    for j in range(i, i + transition_length):
                        if random.random() < (j - i) / transition_length:
                            yawn_rate_curve[j] = next_value
                        else:
                            yawn_rate_curve[j] = current_value
    
    # 初始化累计计数和滑动窗口
    blink_count = 0
    yawn_count = 0
    blink_timestamps = deque()
    yawn_timestamps = deque()
    data_records = []

    # 眨眼累积器和计时器
    blink_accumulator = random.uniform(0, 0.5)
    time_since_last_blink = random.randint(0, 2)

    # 哈欠和点头状态控制
    yawning = False
    yawn_duration = 0
    nodding = False
    nod_duration = 0

    # 头部下垂和眼睛闭合的状态及持续时间
    head_down_state = False
    head_down_duration = 0
    eyes_closed_state = False
    eyes_closed_duration = 0

    # 持续几秒的状态变化阈值
    head_down_threshold = 3
    eyes_closed_threshold = 2

    # 为每个时间点生成数据
    for i in range(num_records):
        current_time = timestamps[i]
        
        # 更新滑动窗口
        while blink_timestamps and (current_time - blink_timestamps[0]).total_seconds() > 60:
            blink_timestamps.popleft()
        
        while yawn_timestamps and (current_time - yawn_timestamps[0]).total_seconds() > 3600:
            yawn_timestamps.popleft()
        
        # 获取当前目标频率
        target_blink_rate = blink_rate_curve[i]
        target_yawn_rate = yawn_rate_curve[i]
        
        # 更新眨眼行为
        time_since_last_blink += 1
        
        # 计算眨眼阈值
        blink_base_threshold = 60 / target_blink_rate
        random_factor = random.normalvariate(1.0, 0.15)
        blink_threshold = max(0.5, blink_base_threshold * random_factor)
        
        # 累积器模拟眨眼
        accumulation_rate = 1 / blink_threshold
        accumulation_rate *= random.uniform(0.95, 1.05)
        blink_accumulator += accumulation_rate
        
        # 决定本秒是否眨眼
        interval_blinks = 0
        if time_since_last_blink >= blink_threshold or blink_accumulator >= 1.0:
            interval_blinks = 1
            blink_count += interval_blinks
            blink_accumulator = random.uniform(0, 0.2)
            time_since_last_blink = 0
            blink_timestamps.append(current_time)

        # 更新哈欠状态
        if yawning:
            yawn_duration -= 1
            if yawn_duration <= 0:
                yawning = False
        else:
            base_probability = target_yawn_rate / 3600
            
            if random.random() < base_probability:
                yawning = True
                yawn_duration = random.randint(2, 6)
                yawn_count += 1
                yawn_timestamps.append(current_time)

        # 更新点头状态
        if nodding:
            nod_duration -= 1
            if nod_duration <= 0:
                nodding = False
        else:
            # 计算点头概率
            if target_blink_rate < 15:
                nod_probability = 0.00005 * random.uniform(0.8, 1.2)
            elif target_blink_rate < 25:
                nod_probability = 0.0002 * random.uniform(0.85, 1.15)
            elif target_blink_rate < 30:
                nod_probability = 0.0005 * random.uniform(0.9, 1.1)
            else:
                nod_probability = 0.001 * random.uniform(0.95, 1.05)
                
            if random.random() < nod_probability:
                nodding = True
                nod_duration = random.randint(1, 3)

        # 更新头部下垂状态
        if head_down_state:
            head_down_duration += 1
            if random.random() < (0.1 * random.uniform(0.8, 1.2)):
                head_down_state = False
                head_down_duration = 0
        else:
            # 头部下垂概率随疲劳增加
            base_head_down_prob = 0
            if target_blink_rate < 15:
                base_head_down_prob = 0.0001
            elif target_blink_rate < 25:
                base_head_down_prob = 0.0003
            elif target_blink_rate < 30:
                base_head_down_prob = 0.0005
            else:
                base_head_down_prob = 0.001
            
            head_down_change_prob = base_head_down_prob * random.uniform(0.85, 1.15)
                
            if random.random() < head_down_change_prob:
                head_down_state = True
                head_down_duration = 1
            else:
                head_down_duration = 0

        # 持续头部下垂判定
        head_down = 1 if head_down_duration >= head_down_threshold else 0

        # 更新眼睛闭合状态
        if eyes_closed_state:
            eyes_closed_duration += 1
            if random.random() < (0.2 * random.uniform(0.9, 1.1)):
                eyes_closed_state = False
                eyes_closed_duration = 0
        else:
            # 眼睛闭合概率随疲劳增加
            base_eyes_closed_prob = 0
            if target_blink_rate < 15:
                base_eyes_closed_prob = 0.0001
            elif target_blink_rate < 25:
                base_eyes_closed_prob = 0.0002
            elif target_blink_rate < 30:
                base_eyes_closed_prob = 0.0004
            else:
                base_eyes_closed_prob = 0.0008
            
            eyes_closed_change_prob = base_eyes_closed_prob * random.uniform(0.85, 1.15)
                
            if random.random() < eyes_closed_change_prob:
                eyes_closed_state = True
                eyes_closed_duration = 1
            else:
                eyes_closed_duration = 0

        # 持续眼睛闭合判定
        sustained_eyes_closed = eyes_closed_duration >= eyes_closed_threshold

        # 计算频率
        blink_rate_per_minute = len(blink_timestamps)
        yawn_rate_per_hour = len(yawn_timestamps)
        
        # 处理初始阶段
        elapsed_seconds = i + 1
        if elapsed_seconds < 60:
            blink_rate_per_minute = blink_rate_per_minute * (60 / elapsed_seconds)
        
        if elapsed_seconds < 3600:
            yawn_rate_per_hour = yawn_rate_per_hour * (3600 / elapsed_seconds)
        
        # 添加随机波动
        blink_rate_per_minute = blink_rate_per_minute * random.uniform(0.98, 1.02)
        yawn_rate_per_hour = round(yawn_rate_per_hour)

        # 在生成数据时动态调整相关性
        current_correlation = get_dynamic_correlation(base_fatigue_curve[i])
        
        # 在重度疲劳状态下同步变化两个指标
        if blink_rate_per_minute > 28:  # 接近重度疲劳阈值
            # 确保哈欠率也相应提高
            yawn_rate_adjustment = (blink_rate_per_minute - 28) / 10
            yawn_rate_per_hour = max(yawn_rate_per_hour, 3 + yawn_rate_adjustment * 5)
        
        if yawn_rate_per_hour > 4:  # 接近重度疲劳阈值
            # 确保眨眼率也相应提高
            blink_rate_adjustment = (yawn_rate_per_hour - 4) / 3
            blink_rate_per_minute = max(blink_rate_per_minute, 25 + blink_rate_adjustment * 10)
        
        # 当一个指标达到重度疲劳时，另一个指标也应该接近重度疲劳
        if blink_rate_per_minute > 30:
            yawn_rate_per_hour = max(yawn_rate_per_hour, 4)  # 确保至少中度疲劳
            
            # 90%的概率让哈欠频率也达到重度疲劳
            if random.random() < 0.9 and base_fatigue_curve[i] > 0.8:
                yawn_rate_per_hour = max(yawn_rate_per_hour, 6)  # 重度疲劳
        
        if yawn_rate_per_hour > 5:
            blink_rate_per_minute = max(blink_rate_per_minute, 28)  # 确保至少接近重度疲劳
            
            # 90%的概率让眨眼频率也达到重度疲劳
            if random.random() < 0.9 and base_fatigue_curve[i] > 0.8:
                blink_rate_per_minute = max(blink_rate_per_minute, 32)  # 重度疲劳
        
        # 确保在危险状态下两个指标都达到最大值
        if head_down and sustained_eyes_closed:
            blink_rate_per_minute = max(blink_rate_per_minute, 35)
            yawn_rate_per_hour = max(yawn_rate_per_hour, 8)

        # 计算疲劳等级和分数
        fatigue_level, fatigue_score = calculate_fatigue_level(
            blink_rate_per_minute, 
            yawn_rate_per_hour,
            head_down,
            sustained_eyes_closed
        )
        
        # 添加随机波动到疲劳分数
        fatigue_score = fatigue_score * random.uniform(0.99, 1.01)
        fatigue_score = min(100, max(0, fatigue_score))

        # 点头状态
        nod_status = 1 if nodding else 0

        # 添加记录
        data_records.append({
            'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Blink Count': blink_count,
            'Blink Rate(per minute)': round(blink_rate_per_minute, 2),
            'Yawn Count': yawn_count,
            'Yawn Rate(per hour)': int(yawn_rate_per_hour),
            'Nod Status': nod_status,
            'Head Down Status': head_down,
            'Fatigue Score': round(fatigue_score, 2),
            'Fatigue Level': get_fatigue_level_name(fatigue_level)
        })

    # 创建DataFrame
    df = pd.DataFrame(data_records)
    
    # 确保只包含用户要求的字段
    columns = [
        'Timestamp', 
        'Blink Count', 
        'Blink Rate(per minute)', 
        'Yawn Count', 
        'Yawn Rate(per hour)', 
        'Nod Status', 
        'Head Down Status', 
        'Fatigue Score', 
        'Fatigue Level'
    ]
    
    df = df[columns]
    return df


def save_fatigue_data_to_csv(filename='driving_fatigue_data.csv', duration_hours=None):
    """生成连续每秒的虚拟驾驶疲劳数据并保存为CSV文件"""
    df = generate_fatigue_data(duration_hours)
    df.to_csv(filename, index=False)
    
    print(f"已生成{len(df)}条连续每秒的驾驶疲劳数据记录，持续时间约{len(df)/3600:.2f}小时")
    print(f"数据已保存至: {filename}")
    
    # 输出示例数据
    print("\n数据示例:")
    print(df.head(1).to_string(index=False))
    
    return df, filename


def batch_generate_fatigue_data(num_files=1000, output_dir=None):
    """批量生成多份疲劳数据
    
    Args:
        num_files: 要生成的文件数量
        output_dir: 保存文件的目录路径，如果为None则自动创建时间戳目录
    
    Returns:
        生成文件的列表
    """
    import os
    from datetime import datetime
    
    # 如果未指定输出目录，创建带时间戳的目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"fatigue_data_{timestamp}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    total_records = 0
    fatigue_level_stats = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Dangerous": 0}
    
    print(f"开始批量生成{num_files}份疲劳数据...")
    print(f"输出目录: {output_dir}")
    
    for i in range(num_files):
        # 生成随机时长
        duration_hours = random.uniform(4.0, 6.0)
        
        # 文件名中包含序号和随机生成的小时数
        filename = os.path.join(output_dir, f"fatigue_data_{i+1:04d}_{duration_hours:.1f}h.csv")
        
        # 生成数据并保存
        df = generate_fatigue_data(duration_hours)
        df.to_csv(filename, index=False)
        
        # 统计信息
        total_records += len(df)
        level_counts = df['Fatigue Level'].value_counts()
        for level, count in level_counts.items():
            fatigue_level_stats[level] += count
        
        generated_files.append(filename)
        
        # 显示进度
        if (i+1) % 10 == 0 or i+1 == num_files:
            print(f"进度: {i+1}/{num_files} ({(i+1)/num_files*100:.1f}%)")
    
    print(f"\n批量生成完成! 共生成{num_files}份数据文件")
    print(f"总数据点: {total_records}条")
    
    # 显示疲劳等级分布统计
    print("\n总体疲劳等级分布:")
    for level, count in fatigue_level_stats.items():
        if count > 0:  # 只显示有数据的等级
            percentage = count / total_records * 100
            hours = count / 3600
            print(f"{level}: {count}秒 ({percentage:.1f}%) - 约{hours:.1f}小时")
    
    # 生成数据集摘要
    generate_dataset_summary(output_dir)
    
    return generated_files, output_dir


def generate_dataset_summary(output_dir):
    """为生成的数据集创建摘要文件
    
    Args:
        output_dir: 数据文件存储的目录
    """
    import os
    import glob
    import pandas as pd
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    
    # 创建摘要数据
    summary_data = []
    
    for file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 文件基本信息
        file_name = os.path.basename(file)
        record_count = len(df)
        duration_hours = record_count / 3600
        
        # 疲劳等级分布
        level_counts = df['Fatigue Level'].value_counts()
        
        # 生理指标统计
        max_blink_rate = df['Blink Rate(per minute)'].max()
        avg_blink_rate = df['Blink Rate(per minute)'].mean()
        max_yawn_rate = df['Yawn Rate(per hour)'].max()
        avg_yawn_rate = df['Yawn Rate(per hour)'].mean()
        
        # 获取最严重的疲劳等级
        highest_level = "Normal"
        for level in ["Dangerous", "Severe", "Moderate", "Mild"]:
            if level in level_counts and level_counts[level] > 0:
                highest_level = level
                break
        
        # 创建摘要字典
        summary = {
            "文件名": file_name,
            "数据点数": record_count,
            "持续时间(小时)": duration_hours,
            "最高疲劳等级": highest_level,
            "平均眨眼频率": avg_blink_rate,
            "最大眨眼频率": max_blink_rate,
            "平均哈欠频率": avg_yawn_rate,
            "最大哈欠频率": max_yawn_rate,
        }
        
        # 添加各疲劳等级的占比
        for level in ["Normal", "Mild", "Moderate", "Severe", "Dangerous"]:
            count = level_counts.get(level, 0)
            percentage = count / record_count * 100 if record_count > 0 else 0
            summary[f"{level}占比(%)"] = percentage
        
        summary_data.append(summary)
    
    # 创建摘要DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # 保存摘要
    summary_path = os.path.join(output_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"数据集摘要已保存至: {summary_path}")
    
    return summary_path


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        try:
            num_files = int(sys.argv[1])
        except ValueError:
            print(f"错误: 文件数量必须为整数，已收到: {sys.argv[1]}")
            sys.exit(1)
    else:
        num_files = 1000
    
    # 检查是否提供了输出目录
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 批量生成数据
    output_files, output_dir = batch_generate_fatigue_data(num_files=num_files, output_dir=output_dir)

    # 生成1000个文件到指定目录
    output_files, output_dir = batch_generate_fatigue_data(
        num_files=1000, 
        output_dir="D:/Project/guaduation_project/data/fatigue_dataset"
    )