from ..config import *
import scipy.io as sio
import os
from PIL import Image, ImageDraw
#run conda install matplotlib to resolve prob as python is not a framework
import matplotlib.pyplot as plt
import shutil



class pre_process:
    '''
    Class pre_process is used to do preprocessing on image
    Including load boundary of well, crop into well and print coordinates.
    '''
    def __init__(self,index=0):
        self.index = index
        self.filename = f'a{index}.jpeg'
        
    def delete_save_folder(self):
        shutil.rmtree(SAVE_PATH)
        os.makedirs(SAVE_PATH)

    def load_entire_image(self):
        img = Image.open(os.path.join(ENTIRE_IMAGE_PATH,self.filename),'r').convert('RGB')
        return img
        # img = Image.open(entire_path,'r').convert('RGB')

    def load_boundary(self):
        boundary = sio.loadmat(BOUNDRY_PATH)['mwBoundary'][0]
        a = boundary[2]
        b = boundary[0]
        c = boundary[3]
        d = boundary[1]
        return([a,b,c,d])

    def generate_well(self,save=False):
        entire_image = self.load_entire_image()
        boundary = self.load_boundary()
        well = entire_image.crop(boundary)
        if save:
            well.save(os.path.join(SAVE_PATH,'well.png'))
        return well


    def load_coordinate(self):
        det = sio.loadmat(COORDINATE_PATH)
        table = det['detection'][0][self.index-1]
        arrayX = []
        arrayY = []
        for i in range(table.shape[0]):
            arrayX.append(table[i][0]-5)
            arrayY.append(table[i][1])
        self.x = arrayX
        self.y = arrayY
        return arrayX,arrayY

    def print_coordinate(self):
        self.delete_save_folder()
        #TODO: use True or save=True??
        well = self.generate_well(True)
        draw = ImageDraw.Draw(well)
        self.load_coordinate()
        X = self.x
        Y = self.y
        for i in range(len(X)):
            draw.ellipse((X[i]-1,Y[i]-1,X[i]+1,Y[i]+1),fill = 'yellow', outline ='yellow')
        well.save(os.path.join(SAVE_PATH,'coordinate.png'))
        
    