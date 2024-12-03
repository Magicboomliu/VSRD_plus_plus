import torch
import logging
import pdb
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


# from config import cfg2 as cfg
from config import cfg3 as cfg
from data import make_data_loader
from solver import build_optimizer, build_scheduler

from utils.check_point import DetectronCheckpointer
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from utils import comm
from utils.backup_files import sync_root

from engine.trainer import do_train
from engine.test_net import run_test

from model.detector import KeypointDetector
from data import build_test_loader
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.backends.cudnn.enabled = True # enable cudnn and uncertainty imported
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # enable cudnn to search the best algorithm


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth 
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup(args)

    distributed = comm.get_world_size() > 1
    if not distributed: cfg.MODEL.USE_SYNC_BN = False

    model = KeypointDetector(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    print("Define the Models Here!")

    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    

    run_test(cfg, checkpointer.model, vis=args.vis, eval_score_iou=args.eval_score_iou, eval_all_depths=args.eval_all_depths)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )