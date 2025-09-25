import os
import cv2
import re
import numpy as np

def natural_sort_key(s):
    """
    自然排序键函数，用于按数字顺序排序文件名
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def images_to_video(folder_path, output_file="output_video.mp4", fps=24, resolution=None):
    """
    将文件夹中的图片按顺序合成为视频
    
    参数:
        folder_path: 包含图片文件的文件夹路径
        output_file: 输出视频文件名 (默认为 output_video.mp4)
        fps: 视频帧率 (默认24帧/秒)
        resolution: 目标分辨率 (宽度, 高度) 或 None 使用第一张图片的分辨率
    """
    # 获取所有图片文件
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file)
    
    if not image_files:
        print("文件夹中没有找到图片文件")
        return
    
    # 按自然顺序排序
    image_files.sort(key=natural_sort_key)
    
    # 确定视频分辨率
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    if first_image is None:
        print(f"无法读取第一张图片: {image_files[0]}")
        return
    
    height, width, _ = first_image.shape
    if resolution:
        width, height = resolution
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print("无法创建视频文件，请检查输出路径和权限")
        return
    
    print(f"开始合成视频: {len(image_files)}张图片 -> {output_file}")
    print(f"分辨率: {width}x{height}, 帧率: {fps}fps")
    
    # 逐帧添加图片
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"跳过无法读取的图片: {image_file}")
            continue
        
        # 调整图片大小到目标分辨率
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        
        # 写入视频帧
        video.write(img)
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"已处理: {i+1}/{len(image_files)} 张图片")
    
    # 释放资源
    video.release()
    print(f"视频合成完成! 已保存至: {output_file}")

if __name__ == "__main__":
    # 用户输入
    folder_path = input("请输入图片文件夹路径: ").strip()
    output_file = input("请输入输出视频文件名 (默认为 output_video.mp4): ").strip()
    fps_input = input("请输入帧率 (默认为24): ").strip()
    resolution_input = input("请输入分辨率 (格式: 宽x高, 如 1920x1080, 默认使用第一张图片分辨率): ").strip()
    
    # 处理默认值
    if not output_file:
        output_file = "output_video.mp4"
    
    try:
        fps = int(fps_input) if fps_input else 24
    except ValueError:
        fps = 24
        print("帧率输入无效，使用默认值24")
    
    resolution = None
    if resolution_input:
        try:
            parts = resolution_input.split('x')
            if len(parts) == 2:
                resolution = (int(parts[0]), int(parts[1]))
        except:
            print("分辨率格式无效，将使用第一张图片的分辨率")
    
    # 执行合成
    images_to_video(folder_path, output_file, fps, resolution)