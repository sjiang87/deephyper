from collections import OrderedDict
from deephyper.benchmarks_hps.params import dropout, optimizer, activation1, activation2, activation3, activation4, activation5
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
            activation1,
            activation2,
            activation3,
            activation4,
            activation5,
            param.discrete("batch_size", 8, 1024, step.GEOMETRIC, 2),
            param.non_ordinal("data_augmentation", [False, True]),
            dropout,
            param.discrete("epochs", 50, 200, step.ARITHMETIC, 1),
            param.discrete("f1_size", 1, 5, step.ARITHMETIC, 1),
            param.discrete("f2_size", 1, 5, step.ARITHMETIC, 1),
            param.discrete("f1_units", 8, 64, step.GEOMETRIC, 2),
            param.discrete("f2_units", 8, 64, step.GEOMETRIC, 2),
            param.continuous("nunits", 1, 1000),
            optimizer,
            param.discrete("p_size", 1, 4, step.ARITHMETIC, 1),
            param.non_ordinal("padding_c1", ["same", "valid"]),
            param.non_ordinal("padding_c2", ["same", "valid"]),
            param.non_ordinal("padding_p1", ["same", "valid"]),
            param.non_ordinal("padding_p2", ["same", "valid"]),
            param.discrete("stride1", 1, 4, step.ARITHMETIC, 1),
            param.discrete("stride2", 1, 4, step.ARITHMETIC, 1)
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = ['relu', 32, True, 0.25, 100, 3, 3, 16, 16, 512, 'adam', 2, 'valid', 'valid', 'same', 'same', 1, 1] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))


