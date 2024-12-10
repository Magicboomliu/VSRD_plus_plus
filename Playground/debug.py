import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle


def load_pickle(file_path):
    """
    Load a pickle file and return its content.
    
    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        object: The content of the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        print(f"Error: The file at {file_path} is not a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__=="__main__":
    
    file_path = "/home/zliu/TPAMI25/AutoLabels/sdflabel/test_labels/demo.pkl"
    
    
    data = load_pickle(file_path)
    

    print(data[1])
    
    pass