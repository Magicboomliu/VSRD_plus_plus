import cv2
import os
from glob import glob
import argparse
from tqdm import tqdm

def images_to_video(image_folder, output_path, fps=30):
    # 获取所有图像文件的路径列表，按文件名排序
    image_files = sorted(glob(os.path.join(image_folder, '*.png')))  # 适用于 PNG 格式
    if not image_files:  # 检查文件夹中是否有图像
        print("No images found in the folder.")
        return

    # 读取第一张图像以获取视频尺寸
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 遍历所有图像并写入视频
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    # 释放视频写入器
    video.release()
    print(f"Video saved as {output_path}")


# 主函数，用于解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a sequence of images to a video.")
    parser.add_argument("--result_folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video (default: 30)")

    args = parser.parse_args()
    
    for sequence in tqdm(os.listdir(args.result_folder)):
        
        sequence_dirname = os.path.join(args.result_folder,sequence)

        visualization_types = os.listdir(sequence_dirname)
        
        for vis in tqdm(visualization_types):
            mp4_folder = os.path.join(sequence_dirname,"mp4")
            os.makedirs(mp4_folder,exist_ok=True)
            sub_vis_folder = os.path.join(sequence_dirname,vis)
            saved_mp4_filename = os.path.join(mp4_folder,vis) +".mp4"

            images_to_video(sub_vis_folder, saved_mp4_filename, args.fps)
