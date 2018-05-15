from .model import *
from .pre_process import *
from .generate_dataset import *
from .logistic_model import *
from .linear_model import *


def _print_coordinate(image_number):
    pre = pre_process(image_number)
    pre.print_coordinate()    

def _generate_dataset(body):
    gen = generate_dataset(**body)
    gen.generate_data_balanced()

def _logistic_model(body):
    log = logistic_model(**body)

def _linear_model(body):
    print(body)
    lin = linear_model_class(**body)