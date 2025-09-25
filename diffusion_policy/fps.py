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

# 示例使用
video_path = "/Users/lucywang/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_kbnr48f5phk612_f8ad/msg/video/2025-09/058d2c211fa955c9292fc5e5059e5d6a.mp4"
fps = get_frame_rate(video_path)
print(f"视频帧率: {fps:.2f} FPS")