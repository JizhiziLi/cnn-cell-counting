from .pre_process import *
from .util import * 
import math
from numpy import random

class generate_dataset:
    '''
    Class generate_dataset is used to generate dateset used for training and testing
    '''
    def __init__(self, **kwargs):
        params = dict(kwargs)
        self.crop_length = params['crop_length']
        self.crop_width = params['crop_width']
        self.cell_size = params['cell_size']
        self.crops_number_per_image = params['crops_number_per_image']
        self.image_start = params['images_range'][0]
        self.image_end = params['images_range'][1]+1
        self.save_name = params['save_name']

    def generate_random_crop(self, img):
        img_length = img.size[0]
        img_width = img.size[1]
        random_X = random.randint(0, img_length - self.crop_length)
        random_Y = random.randint(0, img_width - self.crop_width)
        crop_bound = [random_X, random_Y, random_X + self.crop_length, random_Y + self.crop_width]
        crop = img.crop(crop_bound)
        return crop, crop_bound

    def rgbToInteger(self, r, g, b):
        return r*256*256 + g*256 + b
    
    def count_cell(self, bound, index):
        pre = pre_process(index)
        count = 0
        X, Y = pre.load_coordinate()
        for m in range(len(X)):
            if ((bound[0]+self.cell_size<X[m]) and (bound[2]-self.cell_size>X[m]) and (bound[1]+self.cell_size<Y[m]) and (bound[3]-self.cell_size>Y[m])):
                count = count+1
        return count


    def generate_data_balanced(self):
        '''
        This function is used to generate dataset based on size of crops
        number of crops per image, number of images and so on
        '''
        data = []
        label_classifier = []
        label_count = []
        label_true_count_list = []
        crop_list = []

        for num in range(self.image_start, self.image_end):
            pre = pre_process(num)
            well = pre.generate_well()
            label_true_counter = 0
            label_false_counter = 0
            half_number_crop_per_image = math.floor(self.crops_number_per_image/2)
            print(f'We are now processing image number: {num}...')
            # if(num%10==0):
            #     print(f'We are now processing image number: {num}...')
            while(True):
                rgb = []
                crop, crop_bound = self.generate_random_crop(well)
                count = self.count_cell(crop_bound, num)
                for k in range(len(crop.getdata())):
                    rgb.append(self.rgbToInteger(crop.getdata()[k][0],crop.getdata()[k][1],crop.getdata()[k][2]))
                if(count>0 and label_true_counter < half_number_crop_per_image):
                    label_true_counter+=1
                    label = 1
                    data.append(rgb)
                    label_classifier.append(1)
                    label_count.append(count)

                    label_true_count_list.append(count)
                    crop_list.append(crop)

                elif(count==0 and label_false_counter < half_number_crop_per_image):
                    label_false_counter+=1
                    label=0
                    data.append(rgb)
                    label_classifier.append(0)

                elif(label_true_counter == half_number_crop_per_image and label_false_counter == half_number_crop_per_image):
                    break

        util_instance = util()
        util_instance.save_paramsList(self.save_name,[data, label_count])
        util_instance.save_paramsList(self.save_name+'_classifier',[data, label_classifier])
        util_instance.save_paramsList(self.save_name+'_plot',[crop_list, label_true_count_list])
        util_instance.plot_data_and_label(self.save_name+'_plot')
        print(f'Generate balanced data set and save to params folder as: {self.save_name}.')
        print(f'{len(label_true_count_list)} in {(self.image_end-self.image_start+1)*self.crops_number_per_image} have labels.')

        return data, label_classifier, label_count



                




