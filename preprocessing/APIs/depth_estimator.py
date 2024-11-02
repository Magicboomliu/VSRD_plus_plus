import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import sys
sys.path.append("../..")
from preprocessing.disparity_estimation.Leastereo.retrain.LEAStereo import LEAStereo
from preprocessing.disparity_estimation.Leastereo.config_utils.predict_args import obtain_predict_args


class LEAStereoConfig:
    def __init__(self,
                 kitti2015=1,
                 maxdisp=192,
                 crop_height=384,
                 crop_width=1416,
                 data_path='None',
                 test_list="none",
                 save_path='./predict_depth/',
                 fea_num_layer=6,
                 mat_num_layers=12,
                 fea_filter_multiplier=8,
                 fea_block_multiplier=4,
                 fea_step=3,
                 mat_filter_multiplier=8,
                 mat_block_multiplier=4,
                 mat_step=3,
                 net_arch_fea='sceneflow/best/architecture/feature_network_path.npy',
                 cell_arch_fea='sceneflow/best/architecture/feature_genotype.npy',
                 net_arch_mat='sceneflow/best/architecture/matching_network_path.npy',
                 cell_arch_mat='sceneflow/best/architecture/matching_genotype.npy',
                 resume='Kitti15/best/best.pth',
                 model_path=None):
        
        self.kitti2015 = kitti2015
        self.maxdisp = maxdisp
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.data_path = data_path
        self.test_list = test_list
        self.save_path = save_path
        self.fea_num_layers = fea_num_layer
        self.mat_num_layers = mat_num_layers
        self.fea_filter_multiplier = fea_filter_multiplier
        self.fea_block_multiplier = fea_block_multiplier
        self.fea_step = fea_step
        self.mat_filter_multiplier = mat_filter_multiplier
        self.mat_block_multiplier = mat_block_multiplier
        self.mat_step = mat_step
        self.net_arch_fea = net_arch_fea
        self.cell_arch_fea = cell_arch_fea
        self.net_arch_mat = net_arch_mat
        self.cell_arch_mat = cell_arch_mat
        self.resume = resume
        
        self.net_arch_fea = os.path.join(model_path,net_arch_fea)
        self.cell_arch_fea = os.path.join(model_path,cell_arch_fea)
        self.net_arch_mat = os.path.join(model_path,net_arch_mat)
        self.cell_arch_mat = os.path.join(model_path,cell_arch_mat)
        self.resume = os.path.join(model_path,resume)
        
        assert os.path.exists(self.net_arch_fea)
        assert os.path.exists(self.cell_arch_fea)
        assert os.path.exists(self.net_arch_mat)
        assert os.path.exists(self.cell_arch_mat)
        assert os.path.exists(self.resume)
        
def Load_Depth_Model(model_name,device,model_path="/home/zliu/CVPR25_Detection/VSRD-V2/preprocessing/disparity_estimation/Leastereo/run"):
    if model_name=="LEAStereo":
        opt = LEAStereoConfig(model_path=model_path)
    
    model = LEAStereo(opt)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(opt.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    
    return model
    
def Convert_Disparity_to_Depth(disparity,baseline=0.60,focal_length=552.5543):
    depth = (baseline*focal_length)/(disparity + 1e-6)
    depth = torch.clamp(depth,min=0,max=100)
    # depth = np.clip(depth,a_min=0,a_max=100)
    return depth

def pad_image(image_tensor, target_size):

    B, C, H, W = image_tensor.shape
    target_height, target_width = target_size
    

    pad_height = target_height - H
    pad_width = target_width - W
    
    assert pad_height >= 0 and pad_width >= 0
    
    pad_top = pad_height
    pad_left = pad_width

    padded_tensor = torch.nn.functional.pad(image_tensor, (pad_left, 0, pad_top, 0), mode='constant', value=0)

    unpad_info = [pad_top,pad_left]
    
    original_size = [H,W]
    
    return padded_tensor, unpad_info,original_size

def unpad_image(padded_tensor, unpad_info, original_size):

    H, W = original_size
    pad_top = unpad_info[0]
    pad_left = unpad_info[1]
    
    # 使用切片操作移除填充部分
    unpadded_tensor = padded_tensor[:, :, pad_top:pad_top + H, pad_left:pad_left + W]
    
    return unpadded_tensor

def instance_normalize(image_tensor):

    mean = image_tensor.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    std = image_tensor.std(dim=(2, 3), keepdim=True)    # [B, C, 1, 1]


    std = torch.where(std == 0, torch.tensor(1.0).to(image_tensor.device), std)

    normalized_tensor = (image_tensor - mean) / std

    return normalized_tensor



if __name__=="__main__":
    
    from PIL import Image
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    
    

    # Loaded the Model
    pretrained_disparity_path = "/home/zliu/CVPR25_Detection/VSRD-V2/preprocessing/disparity_estimation/Leastereo/run"    
    device = 'cuda:0'
    model_name = "LEAStereo"
    model = Load_Depth_Model(model_name=model_name,device=device,model_path=pretrained_disparity_path)
    
    
    # loaded the data
    left_image_example = "/home/zliu/CVPR25_Detection/VSRD-V2/preprocessing/APIs/Data_Examples/left/0000004391.png"
    right_image_example = "/home/zliu/CVPR25_Detection/VSRD-V2/preprocessing/APIs/Data_Examples/right/0000004391.png"

    left_image_data = np.array(Image.open(left_image_example).convert("RGB"))
    left_image_data_tensor = torch.from_numpy(left_image_data).permute(2,0,1).unsqueeze(0).float()
    left_image_data_tensor = instance_normalize(left_image_data_tensor)
    
    right_image_data = np.array(Image.open(right_image_example).convert("RGB"))
    right_image_data_tensor = torch.from_numpy(right_image_data).permute(2,0,1).unsqueeze(0).float() 
    right_image_data_tensor = instance_normalize(right_image_data_tensor)
    

    # padding the image for fully divided
    left_image_data_tensor,pad_list,original_size = pad_image(left_image_data_tensor,target_size=(384,1416))
    right_image_data_tensor,pad_list,original_size= pad_image(right_image_data_tensor,target_size=(384,1416))
    
    
    # prediction
    with torch.no_grad():
        model.eval()

        input1 = left_image_data_tensor.to(device)
        input2 = right_image_data_tensor.to(device)
    
        with torch.no_grad():        
            prediction = model(input1, input2)
            prediction = prediction.unsqueeze(0)
            
        prediction = unpad_image(prediction,pad_list,original_size=original_size)
        

        




        
    
    
    
    
    
    
    
