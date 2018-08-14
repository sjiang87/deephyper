from deephyper.models.base import _Problem, param, step

# Benchmark specific hyperparameters
space = [
    param.continuous("foo", -10, 10, start=-10)
]

class Problem(_Problem):
    """Problem class for reutersmlp."""
    def __init__(self):
        super(Problem, self).__init__(space)
        return
