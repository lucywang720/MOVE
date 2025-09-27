import cv2

def get_frame_rate(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return None
    
    # 获取帧率（fps）
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 释放资源
    cap.release()
    return fps