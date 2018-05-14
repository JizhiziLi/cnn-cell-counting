from .model import *
from .preprocess import *


def _print_coordinate(image_number):
    print('----ahahah')
    pre = preprocess(image_number)
    pre.print_coordinate()
    # print(imageName)
