from collections import OrderedDict
import random
NDIM = 2

class Problem:
    def __init__(self):
        space = OrderedDict()
        dim = "x1"
        space[dim] = (-5.0, 10.0)
        dim = "x2"
        space[dim] = (0.0, 15.0)
        self.space = space
        self.params = self.space.keys()
        self.starting_point = [random.uniform(0.0, 10.0) for i in range(NDIM)]
