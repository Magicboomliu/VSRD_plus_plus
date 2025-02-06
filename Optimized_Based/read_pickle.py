import numpy as np
import pickle

def read_pickle_path(path):

    with open(path, "rb") as f:
        loaded_data = pickle.load(f)
    
    return loaded_data



if __name__=="__main__":
    
    data = read_pickle_path("/home/zliu/TPAMI25/KITTI360_Benchmarks/Optimized_Based/estimated_dynamic_00.pkl")
    
    print(data)
    
    pass