
from ..config import *
import pickle
import os
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mean_squared_error
import logging
from math import sqrt
from scipy.stats.stats import pearsonr


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
