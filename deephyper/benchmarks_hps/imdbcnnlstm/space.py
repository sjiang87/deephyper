from collections import OrderedDict
from deephyper.benchmarks_hps.params import optimizer, dropout, activation1, activation2
from deephyper.search.models.base import param, step, padding

space = [
    param.discrete('batch_size', 5, 60, step.ARITHMETIC, 1),
    param.discrete('epochs', 1, 4, step.ARITHMETIC, 1), 
    param.continuous('dropout', 0, 0.5),
    activation1,
    activation2,
    padding,
    optimizer,
    param.discrete('filters', 32, 128, step.ARITHMETIC, 1),
    param.discrete('pool_size', 1, 7, step.ARITHMETIC, 1),
    param.discrete('stride', 1, 5, step.ARITHMETIC, 1),
    param.discrete('maxlen', 80, 120, step.ARITHMETIC, 1),
    param.discrete('lstm_output_size', 50, 100, step.ARITHMETIC, 1),
    param.discrete('embedding_size', 32, 256, step.GEOMETRIC, 2),
    param.discrete('kernel_size', 1, 7, step.ARITHMETIC, 1),
    param.non_ordinal('model_path', ['']),
]
