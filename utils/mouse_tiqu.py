import glob
import os

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# 定义眼睛关键点索引
LEFT_EYE_INDICES = [35, 70, 52, 55, 188, 120, 117]  # 左眼索引
RIGHT_EYE_INDICES = [285, 295, 300, 265, 346, 349, 412]  # 右眼索引
# 新增自定义区域索引
CUSTOM_REGION_INDICES = [165, 164, 391, 287, 406, 18, 182, 57]  # 自定义区域索引


def extract_regions_from_image(image_path, output_dir, face_mesh,
                               regions_to_extract=["left_eye", "right_eye", "custom"], show_debug=False):
    """
    从图像中提取指定区域

    参数:
    image_path: 输入图像路径
    output_dir: 输出目录
    face_mesh: MediaPipe人脸网格检测器
    regions_to_extract: 要提取的区域列表
    show_debug: 是否显示调试图像
    """
    # 获取不带路径的文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return False

    # 转换为RGB (MediaPipe需要RGB输入)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # 进行人脸网格检测
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"未在{image_path}中检测到人脸")
        return False

    # 获取第一个检测到的人脸
    landmarks = results.multi_face_landmarks[0].landmark

    # 用于调试的图像
    debug_image = image.copy() if show_debug else None

    # 提取并保存各个区域
    success = True

    # 提取左眼
    if "left_eye" in regions_to_extract:
        output_path = os.path.join(output_dir, f"{base_name}_left_eye.jpg")
        left_eye_points = np.array([(int(landmarks[idx].x * width),
                                     int(landmarks[idx].y * height))
                                    for idx in LEFT_EYE_INDICES])

        # 计算左眼的边界框
        x_min, y_min = np.min(left_eye_points, axis=0)
        x_max, y_max = np.max(left_eye_points, axis=0)

        # 添加边距
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        # 提取区域
        region_img = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(output_path, region_img)

        if show_debug:
            cv2.polylines(debug_image, [left_eye_points], True, (0, 255, 0), 1)
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

    # 提取右眼
    if "right_eye" in regions_to_extract:
        output_path = os.path.join(output_dir, f"{base_name}_right_eye.jpg")
        right_eye_points = np.array([(int(landmarks[idx].x * width),
                                      int(landmarks[idx].y * height))
                                     for idx in RIGHT_EYE_INDICES])

        # 计算右眼的边界框
        x_min, y_min = np.min(right_eye_points, axis=0)
        x_max, y_max = np.max(right_eye_points, axis=0)

        # 添加边距
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        # 提取区域
        region_img = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(output_path, region_img)

        if show_debug:
            cv2.polylines(debug_image, [right_eye_points], True, (0, 255, 0), 1)
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

    # 提取自定义区域
    if "custom" in regions_to_extract:
        output_path = os.path.join(output_dir, f"{base_name}_custom_region.jpg")
        custom_points = np.array([(int(landmarks[idx].x * width),
                                   int(landmarks[idx].y * height))
                                  for idx in CUSTOM_REGION_INDICES])

        # 计算自定义区域的边界框
        x_min, y_min = np.min(custom_points, axis=0)
        x_max, y_max = np.max(custom_points, axis=0)

        # 添加边距
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        # 提取区域
        region_img = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(output_path, region_img)

        if show_debug:
            cv2.polylines(debug_image, [custom_points], True, (0, 255, 255), 1)  # 自定义区域用黄色标注
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)

    if show_debug:
        cv2.imshow("Detected Regions", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def process_folder(input_folder, output_folder, regions=["custom"], file_extension="jpg", debug=False):
    """
    处理指定文件夹中的所有图像文件

    参数:
    input_folder: 输入图像文件夹路径
    output_folder: 输出图像的文件夹路径
    regions: 要提取的区域列表，可以是 "left_eye", "right_eye", "custom" 的任意组合
    file_extension: 要处理的文件扩展名 (默认为jpg)
    debug: 是否显示调试图像
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 初始化MediaPipe人脸网格检测器
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # 获取所有指定扩展名的图像文件
    pattern = os.path.join(input_folder, f"*.{file_extension}")
    image_files = glob.glob(pattern)

    if not image_files:
        print(f"在{input_folder}中没有找到{file_extension}格式的图像文件")
        return

    print(f"找到{len(image_files)}个{file_extension}文件")
    print(f"将提取以下区域: {', '.join(regions)}")

    # 统计成功和失败的图像数量
    success_count = 0
    fail_count = 0

    # 使用tqdm显示进度条
    for image_path in tqdm(image_files, desc="处理图像"):
        try:
            result = extract_regions_from_image(image_path, output_folder, face_mesh, regions_to_extract=regions,
                                                show_debug=debug)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"处理{image_path}时出错: {str(e)}")
            fail_count += 1

    # 关闭MediaPipe
    face_mesh.close()

    print("\n处理完成!")
    print(f"成功提取: {success_count}张图像")
    print(f"失败: {fail_count}张图像")
    print(f"区域图像已保存到: {output_folder}")


if __name__ == "__main__":
    # 设置输入和输出文件夹
    input_folder = r"C:\Users\woaiy\OneDrive\Desktop\dataset_new\train\yawn\no_yawn"  # 替换为你的输入图像文件夹
    output_folder = r"C:\Users\woaiy\OneDrive\Desktop\dataset_new\train\yawn\no_yawn1"  # 替换为你期望的输出文件夹

    # 选择要提取的区域 - 默认只提取自定义区域
    # regions_to_extract = ["left_eye", "right_eye", "custom"]  # 提取所有区域
    regions_to_extract = ["custom"]  # 只提取自定义区域

    # 启动批量处理
    process_folder(input_folder, output_folder, regions=regions_to_extract, file_extension="jpg", debug=False)