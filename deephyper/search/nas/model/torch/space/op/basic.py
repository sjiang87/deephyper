from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Operation(nn.Module):
    """Interface of an operation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return NotImplementedError

    def init(self, current_node):
        """Preprocess the current operation.
        """


class Tensor(Operation):
    def __init__(self, tensor, *args, **kwargs):
        self.tensor = tensor

    def __str__(self):
        return str(self.tensor)

    def __call__(self, *args, **kwargs):
        return self.tensor
