import argparse
import os
import pickle
from typing import List

import numpy as np


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Step 2: 在 NuScenes infos 中打上动态/静态标记（gt_is_dynamic）。"
    )
    parser.add_argument(
        "--info_path",
        type=str,
        required=True,
        help="输入的 infos_temporal_*.pkl 路径，比如 ./data/nuscenes/nuscenes_infos_temporal_train.pkl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="",
        help="输出路径（默认在原文件名基础上加 _dyn 标记）。",
    )
    parser.add_argument(
        "--speed_thresh",
        type=float,
        default=0.5,
        help="判定动态的速度阈值（m/s），默认 0.5。",
    )
    parser.add_argument(
        "--dynamic_classes",
        type=str,
        default="car,truck,trailer,bus,construction_vehicle,bicycle,motorcycle,pedestrian",
        help="认为“可能是动态”的类别列表，用逗号分隔。",
    )
    return parser.parse_args()


def compute_is_dynamic(
    names: np.ndarray, velocity: np.ndarray, valid_flag: np.ndarray,
    dynamic_classes: List[str], speed_thresh: float
) -> np.ndarray:
    """根据类别和速度判断每个实例是否为动态物体。

    规则：
      1. 仅考虑 valid_flag 为 True 的实例（即点云中真实存在的目标）；
      2. 类别在 dynamic_classes 中（车、人、自行车等“可能会动”的类别）；
      3. 速度模长 > speed_thresh（超过某个速度阈值才认为是“动”的）。
    """
    num = len(names)
    is_dynamic = np.zeros(num, dtype=bool)  # 默认全部视为静态（False）

    if num == 0:
        return is_dynamic

    # 速度模长（LIDAR 坐标系下的 vx, vy，来自 Step1 的 gt_velocity）
    speed = np.linalg.norm(velocity, axis=-1)

    for i in range(num):
        # 无效标注（没有点的框）直接跳过
        if not valid_flag[i]:
            continue
        # 类别不在“可能动态类”中，比如 traffic_cone/barrier，则永远视为静态
        if names[i] not in dynamic_classes:
            continue
        # 速度为 NaN 的情况直接跳过
        if np.isnan(speed[i]):
            continue
        # 速度大于阈值：标记为动态物体
        if speed[i] > speed_thresh:
            is_dynamic[i] = True

    return is_dynamic


def main():
    args = parse_args()

    info_path = args.info_path          # Step1 生成的 infos_temporal_*.pkl 路径
    out_path = args.out_path            # 输出路径（可选）
    speed_thresh = args.speed_thresh    # 速度阈值
    # 将逗号分隔的字符串解析成类别列表，并去掉多余空格
    dynamic_classes = [c.strip() for c in args.dynamic_classes.split(",") if c.strip()]

    # 如果用户没有显式指定 out_path，则在原文件名基础上添加 _dyn 后缀
    if not out_path:
        dirname, basename = os.path.split(info_path)
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(dirname, f"{name}_dyn{ext}")

    print(f"[Step2] 读取 infos: {info_path}")
    print(f"[Step2] 输出路径: {out_path}")
    print(f"[Step2] 动态类别: {dynamic_classes}")
    print(f"[Step2] 速度阈值: {speed_thresh} m/s")
    

    # 读取 Step1 生成的 pkl（结构一般为 {'infos': [...], 'metadata': {...}}）
    with open(info_path, "rb") as f:
        data = pickle.load(f)
    
    
    infos = data.get("infos", [])
    print(f"[Step2] 总 sample 数: {len(infos)}")
    

    num_with_gt = 0  # 统计有 GT 的 sample 数量
    for idx, info in enumerate(infos):
        # 没有三维标注 / 类别 / 速度信息时，跳过该 sample（例如 test 集）
        if "gt_boxes" not in info or "gt_names" not in info or "gt_velocity" not in info:
            continue

        names = np.asarray(info["gt_names"])        # 每个实例的类别名
        velocity = np.asarray(info["gt_velocity"])  # 每个实例的 (vx, vy)
        # valid_flag 已在 Step1 中给出；若不存在则全部视为有效
        valid_flag = np.asarray(info.get("valid_flag", np.ones(len(names), dtype=bool)))



        # 调用上面的规则函数，得到每个实例是否为“动态”的布尔数组
        is_dynamic = compute_is_dynamic(
            names=names,
            velocity=velocity,
            valid_flag=valid_flag,
            dynamic_classes=dynamic_classes,
            speed_thresh=speed_thresh,
        )
        

        # 把结果写回当前 info，键名为 gt_is_dynamic（形状与 gt_names 一致）
        info["gt_is_dynamic"] = is_dynamic
        num_with_gt += 1

    print(f"[Step2] 已处理含 GT 的 sample 数: {num_with_gt}")

    # 将更新后的 data（包括 infos + metadata）写回新的 pkl 文件
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print("[Step2] 完成。带有 gt_is_dynamic 的 infos 已保存到:", out_path)


if __name__ == "__main__":
    main()


