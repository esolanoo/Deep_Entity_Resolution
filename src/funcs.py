import torch 
import numpy as np
import random
import os    

def set_env():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    set_seed(5338)
    set_deterministic()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_deterministic():
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_device():
    return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

def get_project_root():
    return os.path.abspath(os.getcwd())[:-3]
    # return r"C:\Users\Eduardo\Documents\MIACD\git\MIACD\DeepLearning\Deep_Entity_Resolution"
