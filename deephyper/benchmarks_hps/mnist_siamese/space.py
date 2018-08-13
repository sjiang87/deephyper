from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout, activation1, activation2, activation3
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
                param.discrete('margin', 1, 3, step.ARITHMETIC, 2),
                param.discrete('units', 32, 512, step.ARITHMETIC, 1),
                param.discrete('batch_size', 32, 512, step.ARITHMETIC, 1),
                param.discrete('epochs', 10, 45, step.ARITHMETIC, 1),
                activation1,
                activation2,
                activation3,
                dropout,
                optimizer
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = [1, 128, 128, 20, 'relu', 0.3, 'adam'] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))