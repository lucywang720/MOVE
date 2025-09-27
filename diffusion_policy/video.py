import os
import cv2
import re
import numpy as np

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def images_to_video(folder_path, output_file="output_video.mp4", fps=24, resolution=None):
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file)
    
    if not image_files:
        return
    

    image_files.sort(key=natural_sort_key)
    

    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    if first_image is None:
        return
    
    height, width, _ = first_image.shape
    if resolution:
        width, height = resolution
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not video.isOpened():
        return
    

    for i, image_file in enumerate(image_files):
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        
        video.write(img)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            pass
    
    video.release()
