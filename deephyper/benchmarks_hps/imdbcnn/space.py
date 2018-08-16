from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout, activation1, activation2, padding
from deephyper.search.models.base import param, step
class Problem():
    def __init__(self):
        self.space = [
            activation1,
            activation2,
            param.discrete("batch_size", 8, 128, step.GEOMETRIC, 2),
            dropout,
            param.discrete("embedding_dims", 30, 70, step.ARITHMETIC, 1),
            param.discrete("epochs", 1, 8, step.ARITHMETIC, 1),
            param.discrete("filters", 100, 400, step.ARITHMETIC, 1),
            param.discrete("hidden_dims", 10, 40, step.ARITHMETIC, 1),
            param.discrete("kernel_size", 1, 5, step.ARITHMETIC, 1),
            param.discrete("max_features", 2000, 8000, step.ARITHMETIC, 1),
            param.discrete("maxlen", 200, 506000, step.ARITHMETIC, 1),
            optimizer,
            padding,
            param.discrete("strides", 1, 4, step.ARITHMETIC, 1)
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = ['relu', 32, 0.2, 50, 3, 250, 25, 3, 5000, 400, 'adam', 'same', 1] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))
