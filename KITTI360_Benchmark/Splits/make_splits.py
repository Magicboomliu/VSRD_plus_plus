import torch
import numpy as np
import os


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def start_with_data2d(string):
    return string.index("data_2d")
    

def save_into_txt(content_list,saved_name):
    with open(saved_name,'w') as f:
        for idx, line in enumerate(content_list):
            if idx!=len(content_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)
    



if __name__=="__main__":
    
    # vsrd
    vsrd_path = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits"
    vsrd_training_val_sync = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits/training_sync_file.txt"
    vsrd_testing_sync = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits/testing_sync_file.txt"
    
    vsrd_training_contents = read_text_lines(vsrd_training_val_sync)
    vsrd_testing_contents = read_text_lines(vsrd_testing_sync)
    
    vsrd_sync_trainlist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits/ImageSets/train.txt")
    vsrd_sync_vallist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits/ImageSets/val.txt")
    vsrd_sync_trainvallist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits/ImageSets/trainval.txt")
    vsrd_sync_testlist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrd_splits/ImageSets/test.txt")
    

    trainval_dict = dict()
    for idx, line in enumerate(vsrd_training_contents):    
        contents = line.split()
        original_left_image, synced_left_image, original_right_image, synced_right_image, original_predicted_txt,synced_predicted_txt = contents
        trainval_dict[os.path.basename(synced_left_image)[:-4]] = original_left_image[start_with_data2d(original_left_image):]
    assert len(trainval_dict) == len(vsrd_sync_trainvallist)
    
    test_dict = dict()
    for idx, line in enumerate(vsrd_testing_contents):
        contents = line.split()
        original_left_image, synced_left_image, original_right_image, synced_right_image, original_predicted_txt,synced_predicted_txt = contents
        test_dict[os.path.basename(synced_left_image)[:-4]] = original_left_image[start_with_data2d(original_left_image):]
    assert len(test_dict) == len(vsrd_testing_contents)
    
    
    
    vsrd_original_trainlist = []
    vsrd_original_vallist = []
    vsrd_original_trainvallist = []
    vsrd_original_testlist = []
    
    for idx,fname in enumerate(vsrd_sync_trainlist):
        assert fname in trainval_dict.keys()
        vsrd_original_trainlist.append(trainval_dict[fname])
            
    for idx, fname in enumerate(vsrd_sync_vallist):
        assert fname in trainval_dict.keys()
        vsrd_original_vallist.append(trainval_dict[fname])
    
    for idx, fname in enumerate(vsrd_sync_trainvallist):
        assert fname in trainval_dict.keys()
        vsrd_original_trainvallist.append(trainval_dict[fname])

    for idx, fname in enumerate(vsrd_sync_testlist):
        assert fname in test_dict.keys()
        vsrd_original_testlist.append(test_dict[fname])
    
    
    vsrd_original_splits_train = "vsrd_splits/original_ImageSets/ImageSets/train.txt"
    vsrd_original_splits_trainval = "vsrd_splits/original_ImageSets/ImageSets/trainval.txt"
    vsrd_original_splits_val = "vsrd_splits/original_ImageSets/ImageSets/val.txt"
    vsrd_original_splits_test = "vsrd_splits/original_ImageSets/ImageSets/test.txt"
    
    os.makedirs(os.path.dirname(vsrd_original_splits_train),exist_ok=True)
    save_into_txt(vsrd_original_trainlist,vsrd_original_splits_train)
    save_into_txt(vsrd_original_vallist,vsrd_original_splits_val)
    save_into_txt(vsrd_original_trainvallist,vsrd_original_splits_trainval)
    save_into_txt(vsrd_original_testlist,vsrd_original_splits_test)

    #----------------------------------------------------------------------------------------------------------------------------------------#
    
    
    
    # vsrd pp
    vsrd_path = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist"
    vsrd_training_val_sync = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/training_sync_file.txt"
    vsrd_testing_sync = "/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/testing_sync_file.txt"
    
    vsrd_training_contents = read_text_lines(vsrd_training_val_sync)
    vsrd_testing_contents = read_text_lines(vsrd_testing_sync)
    
    vsrd_sync_trainlist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/ImageSets/train.txt")
    vsrd_sync_vallist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/ImageSets/val.txt")
    vsrd_sync_trainvallist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/ImageSets/trainval.txt")
    vsrd_sync_testlist = read_text_lines("/home/zliu/TPAMI25/KITTI360_Benchmarks/KITTI360_Benchmark/Splits/vsrdpp_splist/ImageSets/test.txt")
    

    trainval_dict = dict()
    for idx, line in enumerate(vsrd_training_contents):    
        contents = line.split()
        original_left_image, synced_left_image, original_right_image, synced_right_image, original_predicted_txt,synced_predicted_txt = contents
        trainval_dict[os.path.basename(synced_left_image)[:-4]] = original_left_image[start_with_data2d(original_left_image):]
    assert len(trainval_dict) == len(vsrd_sync_trainvallist)
    
    test_dict = dict()
    for idx, line in enumerate(vsrd_testing_contents):
        contents = line.split()
        original_left_image, synced_left_image, original_right_image, synced_right_image, original_predicted_txt,synced_predicted_txt = contents
        test_dict[os.path.basename(synced_left_image)[:-4]] = original_left_image[start_with_data2d(original_left_image):]
    assert len(test_dict) == len(vsrd_testing_contents)
    
    
    
    vsrd_original_trainlist = []
    vsrd_original_vallist = []
    vsrd_original_trainvallist = []
    vsrd_original_testlist = []
    
    for idx,fname in enumerate(vsrd_sync_trainlist):
        assert fname in trainval_dict.keys()
        vsrd_original_trainlist.append(trainval_dict[fname])
            
    for idx, fname in enumerate(vsrd_sync_vallist):
        assert fname in trainval_dict.keys()
        vsrd_original_vallist.append(trainval_dict[fname])
    
    for idx, fname in enumerate(vsrd_sync_trainvallist):
        assert fname in trainval_dict.keys()
        vsrd_original_trainvallist.append(trainval_dict[fname])

    for idx, fname in enumerate(vsrd_sync_testlist):
        assert fname in test_dict.keys()
        vsrd_original_testlist.append(test_dict[fname])
    
    
    vsrd_original_splits_train = "vsrdpp_splist/original_ImageSets/ImageSets/train.txt"
    vsrd_original_splits_trainval = "vsrdpp_splist/original_ImageSets/ImageSets/trainval.txt"
    vsrd_original_splits_val = "vsrdpp_splist/original_ImageSets/ImageSets/val.txt"
    vsrd_original_splits_test = "vsrdpp_splist/original_ImageSets/ImageSets/test.txt"
    
    os.makedirs(os.path.dirname(vsrd_original_splits_train),exist_ok=True)
    save_into_txt(vsrd_original_trainlist,vsrd_original_splits_train)
    save_into_txt(vsrd_original_vallist,vsrd_original_splits_val)
    save_into_txt(vsrd_original_trainvallist,vsrd_original_splits_trainval)
    save_into_txt(vsrd_original_testlist,vsrd_original_splits_test)
    
    
    
    
    pass

