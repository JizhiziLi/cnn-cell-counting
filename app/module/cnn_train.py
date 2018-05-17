from .cnn_train_layers import *
from .util import *
from ..config import *
import os
import sys
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from pydoc import help
from scipy.stats.stats import pearsonr
import math
from time import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class cnn_train():
    def __init__(self, **kwargs):
        params = dict(kwargs)
        self.width = params['width']
        self.height = params['height']
        self.patience = params['patience']
        self.choice = params['choice']
        self.number = params['number']
        self.cell_number = params['cell_number']
        self.learning_rate = params['learning_rate']
        self.train_set_file = params['train_set_file']
        self.path = CNN_MODEL_PATH
        logging_file = os.path.join(self.path, 'cnn_model_train.log')
        fh = logging.FileHandler(logging_file)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.info('\n\n')
        logger.info(
            f'=======CNN Model: run at {datetime.now().isoformat()}=======')
        logger.info(f'Params here as: {params}')
        logger.info('\n')

    def train(self):
        start = time()
        fullyOutputNumber = 1000
        n_epochs = 200
        # nkerns: number of kernels in each layer
        nkerns = [5, 10]
        # settinghere
        layer1_conv = 5
        layer2_conv = 5
        batch_size = 40

        util_instance = util()
        train_set_data = util_instance.load_paramsList(self.train_set_file)[0]
        train_set_label = util_instance.load_paramsList(
            self.train_set_file+'_classifier')[1]
        train_set_count_label = util_instance.load_paramsList(self.train_set_file)[
            1]
        logger.info(f'Size of input is: {str(len(train_set_data))} / {str(len(train_set_label))} / {str(len(train_set_count_label))}')
        # Initial parameter
        rng = numpy.random.RandomState(23455)
        # Load data
        if(self.choice == 'logistic_zeroOne'):
            datasets = util_instance.load_data_for_cnn_train(train_set_data, train_set_label)
        elif(self.choice == 'logistic_count'):
            datasets = util_instance.load_data_for_cnn_train(
                train_set_data, train_set_count_label)
        elif(self.choice == 'linear_count'):
            datasets = util_instance.load_data_for_cnn_train(
                train_set_data, train_set_count_label)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # Calculate batch_size for each data set
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        # Define several variables, x as train data, as the input of layer0
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        ######################
        # Build CNN Model:
        # input+layer0(LeNetConvPoolLayer)+layer1(LeNetConvPoolLayer)+layer2(HiddenLayer)+layer3(LogisticRegression)
        ######################
        logger.info('Model is now building...')

        # Reshape matrix of rasterized images of shape (batch_size, 50*50)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (50,50) is the size of  images.
        layer0_input = x.reshape((batch_size, 1, self.width, self.height))
        logger.info(f'type of layer0_input is {type(layer0_input)}')
        logger.info(layer0_input)
        # The first convolutional_maxpooling layer
        # Size after convolutional: (50-5+1 , 50-5+1) = (46, 46)
        # Size after maxpooling: (46/2, 46/2) = (23, 23), ignore the boundary
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 23, 23)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, self.width, self.height),
            filter_shape=(nkerns[0], 1, layer1_conv, layer1_conv),
            poolsize=(2, 2)
        )
        logger.info(f'-----layer 0 output is {layer0.output}')
        # Second convolutional + maxpooling layer, use last layer's output as input, (batch_size, nkerns[0], 23, 23)
        #
        # Size after convolutional: (23-5+1 , 23-5+1) = (19, 19)
        # Size after maxpooling: (19/2, 19/2) = (9,9), ignore the boundary
        # todo: /2 problem
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 9,9)

        # [1]
        width1 = math.floor((self.width-layer1_conv+1)/2)
        height1 = width1

        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], width1, height1),
            filter_shape=(nkerns[1], nkerns[0], layer2_conv, layer2_conv),
            poolsize=(2, 2)
        )
        logger.info(f'-----layer 1 output is {layer1.output}')
        hiddenlayerSize = math.floor((width1-layer2_conv+1)/2)

        # HiddenLayer full-connected layer, the size of input is (batch_size,num_pixels), so each sample will get a one-dimentional vector after layer0 and layer1
        # Output from last layer (batch_size, nkerns[1], 9,9) can be turned to (batch_size,nkerns[1]*9*9), by flatten

        # [2]
        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            # [2]
            n_in=nkerns[1] * hiddenlayerSize*hiddenlayerSize,
            n_out=fullyOutputNumber,  # output number of full-connected layer, defined, can change
            activation=T.tanh
        )
        logger.info(f'-----layer 2 output is {layer2.output}')
        # Classifier Layer
        ###############
        # Define some basic factors in optimization, cost function, train, validation, test model, updating rules(Gradient Descent)
        ###############
        # Cost Function

        if(self.choice == 'logistic_zeroOne'):
            # n_in equals to the output number of full-connected layer，n_out equals to number of classifications.
            layer3 = LogisticRegression(
                input=layer2.output, n_in=fullyOutputNumber, n_out=2)
            cost = layer3.negative_log_likelihood(y)
        elif(self.choice == 'logistic_count'):
            layer3 = LogisticRegression(
                input=layer2.output, n_in=fullyOutputNumber, n_out=self.cell_number+1)
            cost = layer3.negative_log_likelihood(y)
        elif(self.choice == 'linear_count'):
            layer3 = LinearRegression(
                input=layer2.output, n_in=fullyOutputNumber, n_out=1)
            cost = layer3.errors(y)
        #         cost = layer3.linear_likelihood(y,number)

        test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # All parameters
        # [3]
        params = layer3.params + layer2.params + layer1.params + layer0.params
        #params = layer3.params + layer2.params+layer4.params + layer1.params + layer0.params

        # Gradient of each parameter
        grads = T.grad(cost, params)
        # Updating rules
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        train_model = theano.function(
            [index],
            # [4]
            [cost, layer3.p_y_given_x, layer3.W, layer3.b,
             layer3.y_pred, layer2_input, layer2.output, y],
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        ###############
        # Train CNN to find the best parameter
        ###############
        logger.info('Model is now training...')
        patience_increase = 2
        improvement_threshold = 0.99
        validation_frequency = min(n_train_batches, self.patience / 2)

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(math.floor(n_train_batches)):
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if iter % 100 == 0:
                    logger.info('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(math.floor(n_valid_batches))]
                    this_validation_loss = numpy.mean(validation_losses)
                    logger.info('epoch %i, minibatch %i/%i, validation error %f %%' %
                                (epoch, minibatch_index + 1, n_train_batches,
                                 this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                                improvement_threshold:
                            self.patience = max(
                                self.patience, iter * self.patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(math.floor(n_test_batches))
                        ]
                        test_score = numpy.mean(test_losses)
                        logger.info(('     epoch %i, minibatch %i/%i, test error of '
                                     'best model %f %%') %
                                    (epoch, minibatch_index + 1, n_train_batches,
                                     test_score * 100.))
                # logger.info(f'----layer0 params is {layer0.params}')
                # logger.info(f'----layer1 params is {layer1.params}')
                paramsList = [layer0.params, layer1.params,
                              layer2.params, layer3.params]
                if(self.choice == 'logistic_zeroOne'):
                    util_instance.save_params_for_cnn('cnn_logistic_zeroOne_params',
                                              paramsList)  # save parameter
                elif(self.choice == 'logistic_count'):
                    util_instance.save_params_for_cnn('cnn_logistic_count_params',
                                              paramsList)  # save parameter
                elif(self.choice == 'linear_count'):
                    # [5]
                    # save parameter
                    logger.info(f'-----we have paramsList----{paramsList}')
                    util_instance.save_params_for_cnn(
                        'cnn_linear_count_params', paramsList)

                if self.patience <= iter:
                    done_looping = True
                break

        logger.info('Optimization complete.')
        logger.info('Best validation score of %f %% obtained at iteration %i, '
                    'with test performance %f %%' %
                    (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        logger.info('Finish running in {:.2f} seconds.'.format(time()-start))
