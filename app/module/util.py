
from ..config import *
import pickle
import os

class util:
    def __init__(self, save_name, paramsList=None):
        self.save_name = save_name
        self.paramsList = paramsList
        
    def save_paramsList(self):       
        write_file = open(os.path.join(PARAMS_PATH, f'{self.save_name}.pkl'),'wb')  
        pickle.dump(self.paramsList,write_file,-1)
        write_file.close()

    def load_paramsList(self):
        f = open(os.path.join(PARAMS_PATH, f'{self.save_name}.pkl'),'rb')
        paramsList=pickle.load(f)
        f.close()
        return paramsList