import re
import os


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow-Multi-Baseline Images")
    parser.add_argument(
        "--root_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    parser.add_argument(
        "--configs_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.")

    # get the local rank
    args = parser.parse_args()


    return args


if __name__=="__main__":
    args = parse_args()
    
    
    root = "../../Optimized_Based/configs/"
    
    config_list = ["train_config_sequence_00.py","train_config_sequence_02.py","train_config_sequence_03.py","train_config_sequence_04.py",
                   "train_config_sequence_05.py","train_config_sequence_06.py","train_config_sequence_07.py",
                   "train_config_sequence_09.py","train_config_sequence_10.py"]
    
    
    config_list = [os.path.join(root,f) for f in config_list]
    
    for conf_path in config_list:

        # 原配置文件的路径
        config_file_path = conf_path

        # 读取配置文件内容
        with open(config_file_path, 'r') as file:
            config_content = file.read()

        # 使用正则表达式替换路径
        config_content = re.sub(r'/data1/liu/VSRD_PP_Sync', args.root_path, config_content)
        config_content = re.sub(r'/home/zliu/CVPR2025/VSRD-V2/Optimized_Based/configs', args.configs_path, config_content)

        # 将修改后的内容写回到配置文件中
        with open(config_file_path, 'w') as file:
            file.write(config_content)
