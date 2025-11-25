import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow-Multi-Baseline Images")
    parser.add_argument(
        "--soft_linked_folder",
        type=str,
        default="/data/dataset/KITTI/KITTI360_For_Docker",
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--source_root_folder",
        type=str,
        default="/data/dataset/KITTI/KITTI360_For_Upload",
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    # get the local rank
    args = parser.parse_args()


    return args



if __name__ == "__main__":
    
    args = parse_args()
    soft_linked_folder = args.soft_linked_folder
    source_root_folder = args.source_root_folder
    os.makedirs(soft_linked_folder, exist_ok=True)

    # A helper function to create soft links safely
    def create_soft_link(source_folder, target_folder):
        # Ensure source path and target path are absolute
        source_folder = os.path.abspath(source_folder)
        target_folder = os.path.abspath(target_folder)
        
        # Check if the target folder already exists
        if os.path.islink(target_folder) or os.path.exists(target_folder):
            os.remove(target_folder)  # Remove existing soft link or folder
        print(f"Creating soft link from {source_folder} to {target_folder}")
        os.symlink(source_folder, target_folder)

    # List of folders to link
    folders_to_link = [
        "annotations",
        "calibration",
        "data_2d_semantics",
        "data_3d_bboxes",
        "data_poses",
        "image_data/data_2d_raw",
        "pseudo_depth_left_NMRF",
        "IGEVStereoSSL"
    ]

    # Create soft links for each folder
    for folder in folders_to_link:
        
        source_folder_path = os.path.join(source_root_folder, folder)
        target_folder_path = os.path.join(soft_linked_folder, os.path.basename(folder))

        # source_folder_path = os.path.join(source_root_folder, folder)
        # target_folder_path = os.path.join(soft_linked_folder, os.path.basename(folder))
        create_soft_link(source_folder_path, target_folder_path)
