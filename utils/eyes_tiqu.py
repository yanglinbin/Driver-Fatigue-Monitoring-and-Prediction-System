import glob
import os

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# 定义眼睛关键点索引
LEFT_EYE_INDICES = [35, 70, 52, 55, 188, 120, 117]  # 左眼索引
RIGHT_EYE_INDICES = [285, 295, 300, 265, 346, 349, 412]  # 右眼索引


def extract_eyes_from_image(image_path, output_dir, face_mesh, show_debug=False):
    # 获取不带路径的文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 输出文件路径
    output_left_path = os.path.join(output_dir, f"{base_name}_left_eye.jpg")
    output_right_path = os.path.join(output_dir, f"{base_name}_right_eye.jpg")

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

    # 提取左眼坐标
    left_eye_points = np.array([(int(landmarks[idx].x * width),
                                 int(landmarks[idx].y * height))
                                for idx in LEFT_EYE_INDICES])

    # 提取右眼坐标
    right_eye_points = np.array([(int(landmarks[idx].x * width),
                                  int(landmarks[idx].y * height))
                                 for idx in RIGHT_EYE_INDICES])

    # 计算左眼的边界框
    left_x_min, left_y_min = np.min(left_eye_points, axis=0)
    left_x_max, left_y_max = np.max(left_eye_points, axis=0)

    # 计算右眼的边界框
    right_x_min, right_y_min = np.min(right_eye_points, axis=0)
    right_x_max, right_y_max = np.max(right_eye_points, axis=0)

    # 添加边距
    padding = 10
    left_x_min = max(0, left_x_min - padding)
    left_y_min = max(0, left_y_min - padding)
    left_x_max = min(width, left_x_max + padding)
    left_y_max = min(height, left_y_max + padding)

    right_x_min = max(0, right_x_min - padding)
    right_y_min = max(0, right_y_min - padding)
    right_x_max = min(width, right_x_max + padding)
    right_y_max = min(height, right_y_max + padding)

    # 提取左眼区域
    left_eye_img = image[left_y_min:left_y_max, left_x_min:left_x_max]

    # 提取右眼区域
    right_eye_img = image[right_y_min:right_y_max, right_x_min:right_x_max]

    # 保存图像
    cv2.imwrite(output_left_path, left_eye_img)
    cv2.imwrite(output_right_path, right_eye_img)

    if show_debug:
        # 在原图像上绘制眼睛区域并显示
        debug_image = image.copy()
        cv2.polylines(debug_image, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(debug_image, [right_eye_points], True, (0, 255, 0), 1)
        cv2.rectangle(debug_image, (left_x_min, left_y_min), (left_x_max, left_y_max), (255, 0, 0), 1)
        cv2.rectangle(debug_image, (right_x_min, right_y_min), (right_x_max, right_y_max), (255, 0, 0), 1)

        cv2.imshow("Detected Eye Regions", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def process_folder(input_folder, output_folder, file_extension="jpg", debug=False):
    """
    处理指定文件夹中的所有图像文件

    参数:
    input_folder: 输入图像文件夹路径
    output_folder: 输出眼睛图像的文件夹路径
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

    # 统计成功和失败的图像数量
    success_count = 0
    fail_count = 0

    # 使用tqdm显示进度条
    for image_path in tqdm(image_files, desc="处理图像"):
        try:
            result = extract_eyes_from_image(image_path, output_folder, face_mesh, show_debug=debug)
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
    print(f"眼睛图像已保存到: {output_folder}")


if __name__ == "__main__":
    # 设置输入和输出文件夹
    input_folder = r"D:\Project\guaduation_project\test\open_eyes"  # 替换为你的输入图像文件夹
    output_folder = r"D:\Project\guaduation_project\test\open_eyes1"  # 替换为你期望的输出文件夹

    # 启动批量处理
    process_folder(input_folder, output_folder, file_extension="png", debug=False)