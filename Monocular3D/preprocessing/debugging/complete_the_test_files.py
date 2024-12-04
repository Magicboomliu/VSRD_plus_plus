import os
import shutil


if __name__=="__main__":
    
    vsrd_old = "/media/zliu/data12/dataset/CVPR24_VSRD_Results/VSRD24_TXT/predictions/2013_05_28_drive_0010_sync/image_00/data_rect/"
    vsrd_PP = "/media/zliu/data12/dataset/CVPR24_VSRD_Results/VSRDPP_Pesudo_Labels/predictions/2013_05_28_drive_0010_sync/image_00/data_rect/"
    
    missing_items = []
    for fname in os.listdir(vsrd_PP):
        vsrd_pp_filename = os.path.join(vsrd_PP,fname)
        vsrd_old_filename = os.path.join(vsrd_old,fname)
        if not os.path.exists(vsrd_old_filename):
            missing_items.append(vsrd_old_filename)
        
    print(len(missing_items))
    
    pass