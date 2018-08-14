from collections import OrderedDict
from deephyper.benchmarks_hps.params import (activation, dropout, optimizer,
                                         activation_flat_params,
                                         optimizer_flat_params)
from deephyper.search.models.base import param, step, Space

epochs = param.discrete("epochs", 5, 500, step.ARITHMETIC, 1),
batch_size = param.discrete("batch_size", 8, 1024, step.GEOMETRIC, 2),

space = Space([
    activation,
    batch_size,
    dropout,
    epochs,
    optimizer
])

space_flat = Space([
    *activation_flat_params,
    batch_size,
    dropout,
    epochs,
    *optimizer_flat_params
])