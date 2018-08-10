from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
                param.discrete('epochs', 2, 12, step.ARITHMETIC, 1),
                param.discrete('batch_size', 8, 128, step.GEOMETRIC, 2),
                optimizer,
                dropout
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = [5, 32, 'adam', 0.2] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))
