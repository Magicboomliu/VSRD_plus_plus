import argparse



from nuscenes_converter import create_nuscenes_infos




def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: create temporal NuScenes info files (train/val)."
    )
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="NuScenes raw data根目录，例如 /data/nuscenes",
    )
    parser.add_argument(
        "--can_bus_root_path",
        type=str,
        required=True,
        help="NuScenes can_bus 数据根目录，例如 /data/nuscenes",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="输出 info pkl 的目录，例如 ./data/nuscenes",
    )
    parser.add_argument(
        "--info_prefix",
        type=str,
        default="nuscenes",
        help="输出文件前缀（默认 nuscenes）",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"],
        help="NuScenes 版本（默认 v1.0-trainval）",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        default=10,
        help="每个 key-frame 使用的历史 sweeps 数量（默认 10）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    
    

    print(
        f"[Step1] Create NuScenes infos:\n"
        f"  root_path           = {args.root_path}\n"
        f"  can_bus_root_path   = {args.can_bus_root_path}\n"
        f"  out_path            = {args.out_path}\n"
        f"  info_prefix         = {args.info_prefix}\n"
        f"  version             = {args.version}\n"
        f"  max_sweeps          = {args.max_sweeps}"
    )

    create_nuscenes_infos(
        root_path=args.root_path,
        out_path=args.out_path,
        can_bus_root_path=args.can_bus_root_path,
        info_prefix=args.info_prefix,
        version=args.version,
        max_sweeps=args.max_sweeps,
    )

    print("[Step1] Done. Infos saved to", args.out_path)


if __name__ == "__main__":
    main()

