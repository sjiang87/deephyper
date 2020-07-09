from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Operation(nn.Module, ABC):
    """Interface of an operation.

    TODO
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        try:
            return str(self)
        except:
            return type(self).__name__

    @abstractmethod
    def __call__(self, tensors: list, seed: int = None, **kwargs):
        """
        Args:
            tensors (list): a list of incoming tensors.

        Returns:
            tensor: an output tensor.
        """
        raise NotImplementedError

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
