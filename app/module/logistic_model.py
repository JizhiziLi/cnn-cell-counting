from sklearn import linear_model, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split
from .util import *
import logging
from ..config import *
from datetime import datetime
from time import time


logger = logging.getLogger(__name__)


class logistic_model:
    def __init__(self, **kwargs):
        start = time()
        params = dict(kwargs)
        self.path = LOGISTIC_MODEL_PATH

        # run different approach based on 'choice'
        if params['train']['choice'] == 1:
            self.logisticRegressionCV(**params['train'])
        elif params['train']['choice'] == 2:
            self.logisticRegressionSplit(**params['train'])

        if params['test']['choice'] == 1:
            self.testLogisterRegression(**params['test'])

        logger.info('Finish running in {:.2f} seconds.'.format(time()-start))

    def logisticRegressionCV(self, **kwargs):
        util_instance = util()
        params = dict(kwargs)
        data_set = params['data_set']
        train_data = util_instance.load_paramsList(data_set)[0]
        train_label = util_instance.load_paramsList(data_set)[1]

        # set up logging thing
        logging_file = os.path.join(self.path, 'logistic_model.log')
        fh = logging.FileHandler(logging_file)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.info('\n\n')
        logger.info(
            f'=======logistic Regression Model: run at {datetime.now().isoformat()}=======')
        logger.info(f'Params here as: {params}')
        logger.info(f'The size of data is: {len(train_data)}')
        logger.info('\n')
        logger.info(f'--------logisticRegressionCV--------')
        logger.info('Logistic Regression Model')

        # train the model
        logreg = linear_model.LogisticRegression(C=1e5)
        scores = cross_val_score(logreg, train_data, train_label, cv=3)
        logger.info(f'Socres are {scores}')
        logger.info("Accuracy: %0.2f (+/- %0.2f)" %
                    (scores.mean(), scores.std() * 2))
        logger.info('\n')
        self.logreg = logreg
        return self.logreg

    def logisticRegressionSplit(self, **kwargs):
        util_instance = util()
        params = dict(kwargs)
        data_set = params['data_set']
        train_data = util_instance.load_paramsList(data_set)[0]
        train_label = util_instance.load_paramsList(data_set)[1]

        # set up logging thing
        logging_file = os.path.join(self.path, 'logistic_model.log')
        fh = logging.FileHandler(logging_file)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.info('\n\n')
        logger.info(
            f'=======logistic Regression Model: run at {datetime.now().isoformat()}=======')
        logger.info(f'Params here as: {params}')
        logger.info(f'The size of data is: {len(train_data)}')
        logger.info('\n')
        logger.info(f'--------logisticRegressionSplit--------')
        logger.info('Logistic Regression Model')

        # train the model
        logreg = linear_model.LogisticRegression(C=1e5)
        X_train, X_test, y_train, y_test = train_test_split(
            train_data, train_label, test_size=0.2, random_state=0)
        logreg.fit(X_train, y_train)
        y_predicted = logreg.predict(X_test)
        logger.info(metrics.classification_report(y_test, y_predicted))
        logger.info("Confusion matrix")
        logger.info(metrics.confusion_matrix(y_test, y_predicted))
        score = logreg.score(X_test, y_test)
        logger.info("Accuracy: %0.2f" % (score))
        logger.info('- Best parameters after grid search: ')
        logger.info('\n')
        self.logreg = logreg
        return logreg

    def testLogisterRegression(self, **kwargs):
        util_instance = util()
        params = dict(kwargs)
        data_set = params['data_set']
        test_data = util_instance.load_paramsList(data_set)[0]
        test_label = util_instance.load_paramsList(data_set)[1]

        scatterdic = {
            "predict": self.logreg.predict(test_data),
            "true": test_label,
            "title": "Simple Linear Regression",
            "name": "test",
            "path": self.path
        }
        util_instance.scatter_fig(**scatterdic)
