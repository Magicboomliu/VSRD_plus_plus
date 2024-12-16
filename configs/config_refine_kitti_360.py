from yacs.config import CfgNode as CN

_C = CN()

_C.INPUT = CN()
_C.INPUT.KITTI_PATH="/media/zliu/data12/dataset/VSRD_PP_Sync/"
_C.INPUT.SYNCED_KITTI_PATH="/media/zliu/data12/dataset/TPAMI_Stage2/OLD_VSRD24_LABEL/VSRD_SPLIT"
_C.INPUT.CSS_PATH="data/nets/css.pt"
_C.INPUT.DEEPSDF_PATH="data/nets/deepsdf.pt"
_C.INPUT.LABEL_TYPE="gt"
_C.INPUT.MASKRCNN_LABELS_PATH=""
_C.INPUT.DIFF_ANNOS='hard'
_C.INPUT.GRID_DENSITY=40
_C.INPUT.RENDERING_AREA=32
_C.INPUT.FILENAME="/home/zliu/TPAMI25/AutoLabels/SDFlabel/data_preprocssing/all_filenames.txt"



_C.OPTIMIZATION = CN()
_C.OPTIMIZATION.ITERS=60
_C.OPTIMIZATION.POSE_ESTIMATOR='kabsch'
_C.OPTIMIZATION.PRECISION="float16"


_C.VISUALIZATION = CN()
_C.VISUALIZATION.VIZ_TYPE='3d'


_C.LOSS = CN()
_C.LOSS.TWOD_WEIGHT=0.3
_C.LOSS.THREED_WEIGHT=0.5

_C.LABELS= CN()
_C.LABELS.PATH='/media/zliu/data12/dataset/CVPR24_VSRD_Results/Autolabels_For_KITTI360_Version_2/'

