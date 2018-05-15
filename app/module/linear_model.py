from sklearn import datasets, linear_model
# from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split
from .util import *
import logging
from ..config import *
from datetime import datetime
from time import time
import numpy as np


logger = logging.getLogger(__name__)

class linear_model_class:
    def __init__(self, **kwargs):
        start = time()
        params = dict(kwargs)
        self.path = LINEAR_MODEL_PATH
        
        #run different approach based on 'choice'
        if params['train']['choice']==1:
            self.simpleLinearRegression(**params['train'])
            if params['test']['choice']==1:
                self.testLinearRegression(**params['test'])

        logger.info('Finish running in {:.2f} seconds.'.format(time()-start))


    def simpleLinearRegression(self,**kwargs):
        util_instance = util()
        params = dict(kwargs)
        data_set = params['data_set']
        train_data = util_instance.load_paramsList(data_set)[0]
        train_label = util_instance.load_paramsList(data_set)[1]

        # set up logging thing
        logging_file = os.path.join(LINEAR_MODEL_PATH,'linear_model.log')
        fh = logging.FileHandler(logging_file)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.info('\n\n')
        logger.info(f'=======linear Regression Model: run at {datetime.now().isoformat()}=======')
        logger.info(f'Params here as: {params}')
        logger.info(f'The size of data is: {len(train_data)}')
        logger.info('\n')
        logger.info(f'--------linearRegressionSplit--------')
        logger.info('Linear Regression Model')
        
        # train the model
        regr = linear_model.LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=0)
        regr.fit(X_train,y_train)
        y_predicted = regr.predict(X_test)
        logger.info(f'Coefficients: {regr.coef_}')
        # The mean square error
        logger.info("Residual sum of squares: %.2f"
        % np.mean((y_predicted - y_test) ** 2))
        logger.info('Variance score: %.2f' % regr.score(X_test, y_test))
        logger.info('\n')
        self.regr = regr
        return regr

    def testLinearRegression(self,**kwargs):
        util_instance = util()
        params = dict(kwargs)
        data_set = params['data_set']
        test_data = util_instance.load_paramsList(data_set)[0]
        test_label = util_instance.load_paramsList(data_set)[1]

        scatterdic = {
            "predict":self.regr.predict(test_data),
            "true":test_label,
            "title":"Simple Linear Regression",
            "name":"test",
            "path":self.path
        }
        util_instance.scatter_fig(**scatterdic)





    