from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
            param.discrete("batch_size", 8, 64, step.GEOMETRIC, 2),
            dropout,
            param.discrete("embedding_dims", 64, 256, step.GEOMETRIC, 2),
            param.discrete("epochs", 1, 7, step.ARITHMETIC, 1),
            param.discrete("max_features", 10000, 30000, step.ARITHMETIC, 1),
            param.discrete("maxlen", 80, 120, step.ARITHMETIC, 1),
            optimizer,
            param.discrete("units",   32, 128, step.GEOMETRIC, 2)
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = [16, 0.5, 128, 4, 20000, 100, 'adam', 64] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))

