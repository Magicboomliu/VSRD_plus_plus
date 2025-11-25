import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import os
from PIL import Image
sys.path.append("../..")
from preprocessing.optical_flow_estimation.flow_vis import Tensor_Flow_To_Color

from preprocessing.optical_flow_estimation.MeMFlow.core.Networks import build_network
from preprocessing.optical_flow_estimation.MeMFlow.core.utils.utils import InputPadder,forward_interpolate
from preprocessing.optical_flow_estimation.MeMFlow.inference import inference_core_skflow as inference_core
from preprocessing.optical_flow_estimation.MeMFlow.configs.kitti_memflownet import get_cfg


def load_optical_flow_pertrained_model(cfg,device):
    model = build_network(cfg).to(device)
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

    model.eval()
    
    return model


def Load_Optical_Flow_Model(model_name,device,model_path):

    if model_name =="MeMFlow":
        cfg = get_cfg()
        cfg.restore_ckpt = model_path
        
        
        optical_flow_pretrained_model = load_optical_flow_pertrained_model(cfg,device=device)

        processor = inference_core.InferenceCore(optical_flow_pretrained_model, config=cfg)
        
    
    
    return processor,cfg


if __name__=="__main__":
    import matplotlib.pyplot as plt
    

    # Load the Model
    model_name = "MeMFlow"
    device = "cuda:0"
    model_path = "/home/zliu/Desktop/CVPR2025/VSRD-V2/data_pre_processing/Dynamic_Static_Clss_Flow_based/MeMFlow/ckpts/MemFlowNet_kitti.pth"
    optical_processor,cfg = Load_Optical_Flow_Model(model_name=model_name,device=device,model_path=model_path)



    # loaded the data
    prev_image_example = "/home/zliu/CVPR25_Detection/VSRD-V2/preprocessing/APIs/Data_Examples/left/0000004391.png"
    next_image_example = "/home/zliu/CVPR25_Detection/VSRD-V2/preprocessing/APIs/Data_Examples/left/0000004392.png"

    prev_image_data = np.array(Image.open(prev_image_example).convert("RGB"))
    prev_image_data = torch.from_numpy(prev_image_data).permute(2,0,1).unsqueeze(0).float()/255.

    
    next_image_data = np.array(Image.open(next_image_example).convert("RGB"))
    next_image_data = torch.from_numpy(next_image_data).permute(2,0,1).unsqueeze(0).float()/255.
    

    padder = InputPadder(prev_image_data.shape)
    prev_image_data = padder.pad(prev_image_data)
    prev_image_data_normalized = prev_image_data * 2.0 -1.0
    prev_image_data_normalized = prev_image_data_normalized.to(device)
    
    # print(prev_image_data_normalized.max())
    # print(prev_image_data_normalized.min())
    # print(prev_image_data_normalized.mean())


    padder = InputPadder(next_image_data.shape)
    next_image_data = padder.pad(next_image_data)
    next_image_data_normalized = next_image_data * 2.0 -1.0
    next_image_data_normalized  = next_image_data_normalized.to(device)


    
    # Inference Here
    with torch.no_grad():
        flow_prev = None
        
        processed_images_concatenation = torch.cat((prev_image_data_normalized,next_image_data_normalized),dim=0).unsqueeze(0)
        
        with torch.no_grad():
            flow_low, flow_pre = optical_processor.step(processed_images_concatenation, end=False,
                                        add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
            flow_pre = padder.unpad(flow_pre[0]).cpu()
            
        estimated_optical_flow = flow_pre.unsqueeze(0).cpu()
    

    # print(estimated_optical_flow.shape)
    # print(estimated_optical_flow.min())
    # print(estimated_optical_flow.mean())

    # flow_with_color = Tensor_Flow_To_Color(estimated_optical_flow)
    

    
    
    # plt.imshow(flow_with_color)
    # plt.show()
    
    
