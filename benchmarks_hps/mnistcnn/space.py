from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout, activation
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
                activation,
                param.discrete("batch_size", 8, 1024, step.GEOMETRIC, 2),
                dropout,
                param.discrete("epochs", 5, 35, step.ARITHMETIC, 1),
                param.discrete("f1_size", 1, 7, step.ARITHMETIC, 1),
                param.discrete("f2_size", 1, 7, step.ARITHMETIC, 1),
                param.discrete("f1_units", 8, 64, step.GEOMETRIC, 2),
                param.discrete("f2_units", 8, 64, step.GEOMETRIC, 2),
                param.non_ordinal("max_pool", [True, False]),
                param.discrete("nunits", 1, 1000, step.ARITHMETIC, 1),
                optimizer,
                param.non_ordinal("padding_c1", ["valid", "same"]),
                param.non_ordinal("padding_c2", ["valid", "same"]),
                param.non_ordinal("padding_p1", ["valid", "same"]),
                param.non_ordinal("padding_p2", ["valid", "same"]),
                param.discrete('p_size', 1, 5, step.ARITHMETIC, 1)
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = ['relu', 128, 0.3, 12, 3, 3, 16, 16, True, 512, 'adam', 'valid', 'valid', 'same', 'same', 3]

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))