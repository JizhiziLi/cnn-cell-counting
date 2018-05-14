from .model import *
from .pre_process import *
from .generate_dataset import *


def _print_coordinate(image_number):
    pre = pre_process(image_number)
    pre.print_coordinate()    

def _generate_dataset(body):
    gen = generate_dataset(**body)
    gen.generate_data_balanced()