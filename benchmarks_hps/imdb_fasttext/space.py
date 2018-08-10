from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, activation
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
            param.discrete('batch_size',  8, 128, step.GEOMETRIC, 2),
            param.discrete('epochs',  3, 15, step.ARITHMETIC, 1),
            activation,
            param.discrete('max_features',  10000, 30000, step.ARITHMETIC, 1),
            param.discrete('maxlen',  200,600, step.ARITHMETIC, 1),
            param.discrete('embedding_dims',  30, 70, step.ARITHMETIC, 1),
            optimizer
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = [32, 5, 'relu', 20000, 400, 50, 'adam'] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))