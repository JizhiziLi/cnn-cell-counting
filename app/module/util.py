
from ..config import *
import pickle
import os
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mean_squared_error
import logging
from scipy.stats.stats import pearsonr
import numpy
from sklearn.cross_validation import train_test_split
import theano
import theano.tensor as T
from math import sqrt


logger = logging.getLogger(__name__)


class util:
    def __init__(self):
        pass

    def save_paramsList(self, save_name, paramsList):
        write_file = open(os.path.join(PARAMS_PATH, f'{save_name}.pkl'), 'wb')
        pickle.dump(paramsList, write_file, -1)
        write_file.close()

    def load_paramsList(self, save_name):
        f = open(os.path.join(PARAMS_PATH, f'{save_name}.pkl'), 'rb')
        paramsList = pickle.load(f)
        f.close()
        return paramsList

    def scatter_fig(self, **kwargs):
        # load parameter
        params = dict(kwargs)
        predict = params['predict']
        true = params['true']
        title = params['title']
        name = params['name']
        path = params['path']

        # save scatter figure
        pylab.rcParams['figure.figsize'] = (10.0, 10.0)
        img = plt.scatter(true, predict,  color='black')
        plt.ylabel('Predicted Number Of Cells')
        plt.xlabel('True Number Of Cells')
        plt.title(title)
        plt.axis('on')
        fig = plt.gcf()
        fig.savefig(os.path.join(path, name+'.png'), dpi=100)
        # save info to logger
        logging_file = os.path.join(path, name+'.log')
        fh=logging.FileHandler(logging_file)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

        logger.info('MSE:')
        logger.info(mean_squared_error(predict, true))
        logger.info('RMSE:')
        logger.info(sqrt(mean_squared_error(predict, true)))

        logger.info('PEARSON:')
        logger.info(pearsonr(true, predict))

    def load_data_for_cnn_train(self, Data, Label):
        train_data_list,test_data_valid, train_label_list, test_label_valid = train_test_split(Data,Label, test_size=0.2, random_state=0)
        test_data_list,valid_data_list,test_label_list,valid_label_list= train_test_split(test_data_valid,test_label_valid,test_size=0.5,random_state=0)
        train_data = numpy.array(train_data_list)
        train_label = numpy.array(train_label_list)
        valid_data = numpy.array(valid_data_list)
        valid_label = numpy.array(valid_label_list)
        test_data = numpy.array(test_data_list)
        test_label = numpy.array(test_label_list)
        # Data stored as shared type so that they can be copied to GPU and get speed increased.
        def shared_dataset(data_x, data_y, borrow=True):
            shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        train_set_x, train_set_y = shared_dataset(train_data,train_label)
        test_set_x, test_set_y = shared_dataset(test_data,test_label)
        valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)
        rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)]
        return rval

    def load_data_for_cnn_test(self, Data, Label):
        test_data = numpy.array(Data)
        test_label = numpy.array(Label)
        return test_data,test_label

    #Save the parameters from training
    def save_params_for_cnn(self, save_name, paramsList):  
        f = open(os.path.join(PARAMS_PATH, f'{save_name}.pkl'), 'wb')
        for i in range(0,4):
            pickle.dump(paramsList[i], f, -1)   
        f.close()  

    def load_params_for_cnn(self, save_name):
        f = open(os.path.join(PARAMS_PATH, f'{save_name}.pkl'), 'rb')
        layer0_params=pickle.load(f)
        layer1_params=pickle.load(f)
        layer2_params=pickle.load(f)
        layer3_params=pickle.load(f)
        f.close()
        return layer0_params,layer1_params,layer2_params,layer3_params

    def plot_data_and_label(self, data_set):
        data_file = open(os.path.join(PARAMS_PATH, f'{data_set}.pkl'), 'rb')
        paramsList = pickle.load(data_file)
        crop_list = paramsList[0]
        count_list = paramsList[1]
        f = pylab.figure()
        pylab.rcParams['figure.figsize'] = (30,30)
        pylab.axis('off')
        for i in range(20):
            f.add_subplot(4,5,i+1)
            j = numpy.random.randint(0,len(crop_list))
            pylab.imshow(crop_list[j])
            pylab.title("Label: "+str(count_list[j]))
            plt.axis('off')
        plt.axis('off')
        fig = pylab.gcf()
        fig.savefig(SAVE_PATH+'plot.png',bbox_inches='tight')
        