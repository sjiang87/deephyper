from collections import OrderedDict
from deephyper.benchmarks_hps.params import dropout, optimizer, activation
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
            activation,
            param.discrete('batch_size', 8, 128, step.GEOMETRIC, 2),
            param.continuous('base_lr', 1e-3, 1e-1),
            param.discrete('epochs', 100, 300),
            param.continuous('lr80', 1e-3, 1e-1),
            param.continuous('lr120', 1e-4, 1e-1),
            param.continuous('lr160', 1e-4, 1e-2),
            param.continuous('lr180', 1e-5, 1e-2),
            param.discrete('num_filters', 16, 64, step.GEOMETRIC, 2),
            param.discrete('num_filters_in', 16, 64, step.GEOMETRIC, 2),
            optimizer
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = ['relu', 32, 1e-3, 200, 1e-1, 1e-2, 1e-3, 0.5e-3, 16, 16, 'adam'] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))
