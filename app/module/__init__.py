from .model import *
from .pre_process import *
from .generate_dataset import *
from .logistic_model import *
from .linear_model import *
from .cnn_train import *
from .util import *


def _print_coordinate(image_number):
    pre = pre_process(image_number)
    pre.print_coordinate()


def _generate_dataset(body):
    gen = generate_dataset(**body)
    gen.generate_data_balanced()


def _logistic_model(body):
    log = logistic_model(**body)


def _linear_model(body):
    lin = linear_model_class(**body)


def _cnn_model_train(body):
    cnn = cnn_train(**body)
    cnn.train()


def _plot_data_and_label(plot_name):
    util_instance = util()
    util_instance.plot_data_and_label(plot_name)
