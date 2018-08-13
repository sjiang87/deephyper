from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout, activation, activation1, activation2
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
                param.discrete('batch_size', 8, 512, step.GEOMETRIC, 2),
                param.discrete('epochs', 5, 60, step.ARITHMETIC, 1),
                param.discrete('nhidden', 1, 20, step.ARITHMETIC, 1),
                param.discrete('nunits', 1, 1000, step.ARITHMETIC, 1),
                activation,
                activation1,
                activation2,
                dropout,
                optimizer
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = [128, 20, 1, 512, 'relu', 0.2, 'adam'] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))



