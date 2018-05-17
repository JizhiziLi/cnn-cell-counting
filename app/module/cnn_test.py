import os
import sys
import pickle

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import math
import matplotlib.pyplot as plt
from .util import *
from .cnn_test_layers import *
import logging
from datetime import datetime
from ..config import *


logger = logging.getLogger(__name__)


class cnn_test():
    def __init__(self, **kwargs):
        params = dict(kwargs)
        self.test_set_file = params['test_set_file']
        self.params_path = params['params_path']
        self.choice = params['choice']
        self.print_switch = True
        self.path = CNN_MODEL_PATH
        logging_file = os.path.join(self.path, 'cnn_model_test.log')
        fh = logging.FileHandler(logging_file)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.info('\n\n')
        logger.info(
            f'=======CNN Model Test: run at {datetime.now().isoformat()}=======')
        logger.info(f'Params here as: {params}')
        logger.info('\n')

    def test(self):
        fullyOutputNumber = 10
        nkerns = [5, 10]

        util_instance = util()
        test_set_data = util_instance.load_paramsList(self.test_set_file)[0]
        test_set_label = util_instance.load_paramsList(
            self.test_set_file+'_classifier')[1]
        test_set_count_label = util_instance.load_paramsList(self.test_set_file)[
            1]
        data, label = util_instance.load_data_for_cnn_test(
            test_set_data, test_set_count_label)
        data_num = data.shape[0]  # how many data
        logger.info(f'Size for input used in test is {data_num}')
        layer0_params, layer1_params, layer2_params, layer3_params = util_instance.load_params_for_cnn(
            self.params_path)

        x = T.matrix('x')  # used as input for layer0

        ######################
        # Initialise params for all layers, W, b
        ######################
        layer0_input = x.reshape((data_num, 1, 50, 50))
        layer0 = LeNetConvPoolLayer(
            input=layer0_input,
            params_W=layer0_params[0],
            params_b=layer0_params[1],
            image_shape=(data_num, 1, 50, 50),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

    # [3]
        layer1 = LeNetConvPoolLayer(
            input=layer0.output,
            params_W=layer1_params[0],
            params_b=layer1_params[1],
            image_shape=(data_num, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(
            input=layer2_input,
            params_W=layer2_params[0],
            params_b=layer2_params[1],
            n_in=nkerns[1] * 9 * 9,
            n_out=fullyOutputNumber,
            activation=T.tanh
        )

        if(self.choice == 'logistic_zeroOne'):
            layer3 = LogisticRegression(
                input=layer2.output, params_W=layer3_params[0], params_b=layer3_params[1], n_in=fullyOutputNumber, n_out=2)
        elif(self.choice == 'logistic_count'):
            layer3 = LogisticRegression(
                input=layer2.output, params_W=layer3_params[0], params_b=layer3_params[1], n_in=fullyOutputNumber, n_out=16)
        elif(self.choice == "linear_count"):
            layer3 = LinearRegression(
                input=layer2.output, params_W=layer3_params[0], params_b=layer3_params[1], n_in=fullyOutputNumber, n_out=1)

        # Define theno.function, use x as input
        # layer3.y_pred (prediction/classification) is output
        f = theano.function(
            [x],  # input for function is List
            layer3.y_pred
        )

        # pred if the predicted label
        pred = f(data)

        # Print out the wrongly labelled data
        wrongList = []
        # plt.plot(pred,label)

        for i in range(data_num):
            if(label[i] != math.floor(pred[i])):
                wrongList.append(i)
                if(self.print_switch == True):
                    logger.info('picture: %i is %i, mis-predicted as  %i' %
                                (i, label[i], pred[i]))
        logger.info(str(len(wrongList))+" in " +
                    str(len(label))+" has been predicted wrong")
        logger.info("predict accuracy: {:.2f}%".format(
            (len(label)-len(wrongList))/len(label)*100))
        logger.info("CNN test finishes.")
        return label, pred, wrongList
