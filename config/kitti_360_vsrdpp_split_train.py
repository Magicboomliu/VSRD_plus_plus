#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

_C = CN()
_C.TRAIN = CN()
_C.VAL = CN()
_C.INFER = CN()
_C.DATA = CN()

_C.EXP_NAME = "default"
_C.NET_LAYER = 34
_C.RESTORE_PATH = None
_C.RESTORE_EPOCH = None

_C.LOG_DIR = './outputs/kitti360_vsrd_pp_split/logs'
_C.CHECKPOINTS_DIR = './outputs/kitti360_vsrd_pp_split/checkpoints'


_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.EPOCH = 50
_C.TRAIN.LR = 1e-4
_C.TRAIN.WEIGHT_FILE = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/train_weight.txt'
_C.TRAIN.TRAIN_FILE = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/train.txt'
_C.TRAIN.IMAGE_HW = (376, 1408)
_C.TRAIN.SAMPLE_ROI_POINTS = 1000
_C.TRAIN.SAMPLE_LOSS_POINTS = 100
_C.TRAIN.WORKS = 0
_C.TRAIN.FLIP = 0.0

_C.VAL.WORKS = 0
_C.VAL.SPLIT_FILE = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/ImageSets_For_WeakM3D/val.txt'
_C.VAL.GT_DIR = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/label_gt/'

_C.INFER.WORKS = 0
_C.INFER.DET_2D_PATH = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/det2d/' # FIXME
_C.INFER.SAVE_DIR = './outputs/kitti360_vsrd_pp_split/predictions'


_C.DATA.CLS_LIST = ['Car']
_C.DATA.MODE = 'KITTI360'
_C.DATA.ROOT_3D_PATH = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/'
_C.DATA.RoI_POINTS_DIR = 'ROI_LiDAR_InternImage'
_C.DATA.KITTI_360_PATH = '/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_PP_SPLIT/training/'


_C.DATA.TYPE = ['Car']
_C.DATA.IMAGENET_STATS_MEAN = [0.485, 0.456, 0.406]
_C.DATA.IMAGENET_STATS_STD = [0.229, 0.224, 0.225]

_C.DATA.DIM_PRIOR = [[1.6, 1.8, 4.]]


