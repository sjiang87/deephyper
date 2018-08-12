from collections import OrderedDict
from deephyper.search.models.base import param, step
from deephyper.benchmarks_hps.params import optimizer, dropout, activation

class Problem():
    def __init__(self):
        self.space = [
                activation,
                param.discrete("batch_size", 8, 128, step.GEOMETRIC, 2),
                dropout,
                param.discrete("epochs", 2, 20, step.ARITHMETIC, 1),
                param.discrete("max_words", 800, 1300, step.ARITHMETIC, 1),
                param.discrete("nunits", 128, 1024, step.ARITHMETIC, 128),
                optimizer,
                param.discrete("skip_top", 0,30, step.ARITHMETIC, 1)
        ]

        self.params = [param.name for param in self.space]
        self.starting_point = ['relu', 32, 0.5, 5, 1000, 512, 'adam', 0] 

if __name__ == '__main__':
    instance = Problem()
    print(' '.join(f'--{k}={instance.starting_point[i]}' for i,k in enumerate(instance.params)))
