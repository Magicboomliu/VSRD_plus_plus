import pickle
import os
from tqdm import tqdm
import numpy as np



from sklearn.metrics import accuracy_score, recall_score, f1_score

def evaluate_binary_classification(A, B):
    accuracy = accuracy_score(B, A)  # 计算准确率
    recall = recall_score(B, A)  # 计算召回率
    f1 = f1_score(B, A)  # 计算F1-score
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return accuracy, recall, f1

def pickle_file_loaded(file_path):
    with open(file_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def classifications(velocities_tensor,threshold=0.030):
    # velocities tensors is [N,3]
    return_list = []
    for velocities in velocities_tensor:
        velocities_max = np.max(np.abs(velocities))
        if velocities_max>=threshold:
            return_list.append(True)
        else:
            return_list.append(False)
    return return_list


def get_all_filenames(root_folder):
    filenames = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames

def get_the_estimated_results(raw_data):
    est_velo = np.array(raw_data['est_velo']) #(N,3)

    dynamic_gt_velo = raw_data['dynamic_gt']
    
    estimated_labels = classifications(est_velo)
    
    return estimated_labels,dynamic_gt_velo
    



  
if __name__=="__main__":
    
    root_folder = "/media/zliu/data12/TPAMI_Results/EST_VELO/Estimated_Velocites/data_2d_raw/"
    
    GT_LABELS = []
    EST_LABLES = []
    
    fnames = get_all_filenames(root_folder=root_folder)
    

    for fname in tqdm(fnames):
        raw_data = pickle_file_loaded(fname)
        # if "2013_05_28_drive_0007_sync" or "2013_05_28_drive_0003_sync" in fname:        
        estimated_labels,dynamic_gt_velo = get_the_estimated_results(raw_data)
        EST_LABLES.extend(estimated_labels)
        GT_LABELS.extend(dynamic_gt_velo)

    
    accuracy, recall, f1 = evaluate_binary_classification(EST_LABLES,GT_LABELS)

