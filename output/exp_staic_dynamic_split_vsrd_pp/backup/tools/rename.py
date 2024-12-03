import os



if __name__=="__main__":

    val_path = "/home/liuzihua/MonoFlex/output/EXP_pseudo_Results/exp_pseudo_test/inference/kitti_train/data"

    val_path_complete = [os.path.join(val_path,f) for f in  os.listdir(val_path)]

    # print(val_path_complete)

    valid_lines = []
    for idx, test_name in enumerate(val_path_complete):
        saved_test_name = test_name.replace(".ptxt",'.txt')
        saved_test_name = saved_test_name.replace(".pntxt",'.txt')
        saved_test_name = saved_test_name.replace(".pngtxt",'.txt')

        os.rename(test_name,saved_test_name)
        valid_lines.append(saved_test_name)

    pass