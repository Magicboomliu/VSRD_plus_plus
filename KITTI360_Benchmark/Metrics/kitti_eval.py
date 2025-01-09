
import os
import numpy as np
from tqdm import tqdm
from eval import get_official_eval_result


def eval_from_scrach(gt_dir, det_dir, ap_mode=40):
    global AP_mode
    AP_mode = ap_mode

    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    for i, f in enumerate(tqdm(all_f)):
        
        try:
            gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 17)
        except:
            gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 16)
            
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)
        


        gt = {}
        det = {}
        '''bbox'''
        gt['bbox'] = gt_f[:, 4:8].astype(np.float32)
        det['bbox'] = det_f[:, 4:8].astype(np.float32)

        '''alpha'''
        gt['alpha'] = gt_f[:, 3].astype(np.float32)
        det['alpha'] = det_f[:, 3].astype(np.float32)

        '''occluded'''
        gt['occluded'] = gt_f[:, 2].astype(np.float32)
        det['occluded'] = det_f[:, 2].astype(np.float32)

        '''truncated'''
        gt['truncated'] = gt_f[:, 1].astype(np.float32)
        det['truncated'] = det_f[:, 1].astype(np.float32)

        '''name'''
        gt['name'] = gt_f[:, 0]
        det['name'] = det_f[:, 0]

        '''location'''
        gt['location'] = gt_f[:, 11:14].astype(np.float32)
        det['location'] = det_f[:, 11:14].astype(np.float32)

        '''dimensions, convert hwl to lhw'''
        gt['dimensions'] = gt_f[:, [10, 8, 9]].astype(np.float32)
        det['dimensions'] = det_f[:, [10, 8, 9]].astype(np.float32)

        '''rotation_y'''
        gt['rotation_y'] = gt_f[:, 14].astype(np.float32)
        det['rotation_y'] = det_f[:, 14].astype(np.float32)

        '''score'''
        det['score'] = det_f[:, 15].astype(np.float32)

        ''' append to tail'''
        all_gt.append(gt)
        all_det.append(det)

    if AP_mode == 40:
        print('-' * 40 + 'AP40 evaluation' + '-' * 40)
    if AP_mode == 11:
        print('-' * 40 + 'AP11 evaluation' + '-' * 40)

    print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-2]))
    for cls in ['Car']:
        print('*' * 20 + cls + '*' * 20)
        res = get_official_eval_result(all_gt, all_det, cls, z_axis=1, z_center=1)
        Car_res = res['detail'][cls]
        for k in Car_res.keys():
            print(k, Car_res[k])
    print('\n')
    
    

if __name__=='__main__':
    #KITTI3D_Format_DEBUG
    #KITTI3D_VSRDPP_Full
    # KITTI3D_VSRD_Vanilla
    #KITTI3D_Velocity_Only
    
    pd_dir_folder = "/home/zliu/CVPR2025/VSRDPP_Stage2_Experiments/SensetimeJapan_Internship/output/exp_monoflex_update_V2/kitti_train/inference_11008/data"
    gt_dir_folder = "/data1/liu/KITTI360_SoftLink/KITTI360_VSRDPP_V1/training/label_2/"
    # "/data1/liu/KITTI360_SoftLink/KITTI360_VSRDPP_V1/testing/label_2/"
    # "/data1/liu/KITTI360_SoftLink/KITTI360_VSRDPP_V1/training/label_2/"
    
    eval_from_scrach(gt_dir=gt_dir_folder,
                     det_dir=pd_dir_folder,ap_mode=40)

    # eval_from_scrach(gt_dir=gt_dir_folder,
    #                  det_dir=pd_dir_folder,ap_mode=11)
    
    
    
    
    pass