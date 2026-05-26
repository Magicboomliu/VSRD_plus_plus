import numpy as np



def convert_numpy_to_calib_file(cam_intrin_np,output_calib_file):


    P = np.load(cam_intrin_np)[:3,:]
    
    result_list = []
    string_representation_P0 = "P0: "+' '.join(map(str, P.flatten()))
    string_representation_P1 = "P1: "+' '.join(map(str, P.flatten()))
    string_representation_P2 = "P2: "+' '.join(map(str, P.flatten()))
    string_representation_P3 = "P3: "+' '.join(map(str, P.flatten()))
    string_representation_R0_rect = "R0_rect: "+' '.join(map(str, np.ones((3,3)).flatten()))
    string_representation_Tr_velo_to_cam = "Tr_velo_to_cam: "+' '.join(map(str, P.flatten()))
    string_representation_Tr_imu_to_velo = "Tr_imu_to_velo: "+' '.join(map(str, P.flatten()))
    
    result_list.append(string_representation_P0)
    result_list.append(string_representation_P1)
    result_list.append(string_representation_P2)
    result_list.append(string_representation_P3)
    result_list.append(string_representation_R0_rect)
    result_list.append(string_representation_Tr_velo_to_cam)
    result_list.append(string_representation_Tr_imu_to_velo)
    
    
    with open(output_calib_file,'w') as f:
        for idx, line in enumerate(result_list):
            if idx!=len(result_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    

    
    
    


if __name__=="__main__":
    cam_intrin = "/media/zliu/data12/dataset/KITTI/KITTI360/calibration/cam0_intrin.npy"
    
    output_file = "/media/zliu/data12/dataset/VSRD_PP_Sync/cam_calib.txt"
    convert_numpy_to_calib_file(cam_intrin_np=cam_intrin,output_calib_file=output_file)