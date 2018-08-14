from deephyper.models.base import _Problem, param, step
from deephyper.benchmarks_hps.params import (activation_flat_params,
                                             dropout, optimizer_flat_params)

# Benchmark specific hyperparameters
batch_size = param.discrete("batch_size", low=8, high=128,
                             step_type=step.GEOMETRIC, step_size=2,
                             start=128)
epochs = param.discrete("epochs", 2, 20, step.ARITHMETIC, 1, start=5)
max_words = param.discrete("max_words",800, 1300, step.ARITHMETIC, 1,
                           start=1000)
nunits = param.discrete("nunits", 128, 1024, step.ARITHMETIC, 128, start=512)
skip_top = param.discrete("skip_top", 0,30, step.ARITHMETIC, 1, start=0)

space = [
    *activation_flat_params,
    batch_size,
    dropout,
    epochs,
    max_words,
    nunits,
    *optimizer_flat_params,
    skip_top
]

class Problem(_Problem):
    """Problem class for reutersmlp."""
    def __init__(self):
        super(Problem, self).__init__(space)
        return

if __name__ == "__main__":
    instance = Problem()
    print(instance.space)
    print(instance.params)
    print(instance.starting_point)
