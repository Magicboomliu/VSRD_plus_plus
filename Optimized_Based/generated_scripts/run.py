import os
import sys
import time


if __name__=="__main__":


    N_SPLITS = 64
    
    for i in range(N_SPLITS):
        suffix = f"{i:02}"
        scripts_name = "run_script_{}.sh".format(suffix)
        os.system("qsub -g tga-lab_otm {}".format(scripts_name))
        print("----------------------------------")
        time.sleep(2.5)
