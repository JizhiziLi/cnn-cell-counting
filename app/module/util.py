
from ..config import *
import pickle
import os

class util:
    def __init__(self, params_path):
        self.params_path = config.PARAMS_PATH
        
    def save_paramsList(params_file,paramsList):       
        write_file = open(os.path.join(self.params_path, params_file),'wb')  
        pickle.dump(paramsList,write_file,-1)
        write_file.close()

    def load_paramsList(params_file):
        f = open(os.path.join(self.param_path, params_file),'rb')
        paramsList=pickle.load(f)
        f.close()
        return paramsList