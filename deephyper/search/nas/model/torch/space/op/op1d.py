import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Operation


class Dense(Operation):
    """Multi Layer Perceptron operation.

    Help you to create a perceptron with n layers, m units per layer and an activation function.

    Args:
        units (int): number of units per layer.
        activation: an activation function from torch function.
    """

    def __init__(self, units, activation=None, *args, **kwargs):
        super().__init__()

        # Parameters
        self._wT = None
        self._b = None

        # Layer args
        self.units = units
        self.activation = activation
        self.kwargs = kwargs

        # Reuse arg
        self._module = None

    def __str__(self):
        if isinstance(self.activation, str):
            return f"Dense_{self.units}_{self.activation}"
        elif self.activation is None:
            return f"Dense_{self.units}"
        else:
            return f"Dense_{self.units}_{self.activation.__name__}"

    def __call__(self, inputs, seed=None, **kwargs):
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."

        if self._wT is None:  # reuse mechanism
            shape = list(inputs[0].shape)
            self._wT = nn.Parameter(torch.randn(self.units, shape[1]))
            self._b = nn.Parameter(torch.randn(self.units))

        out = F.linear(inputs[0], weight=self._wT, bias=self._b)
        if self.activation is not None:  # better for visualisation
            out = self.activation(out)
        return out


# class Dropout(Operation):
#     """Dropout operation.

#     Help you to create a dropout operation.

#     Args:
#         rate (float): rate of deactivated inputs.
#     """

#     def __init__(self, rate):
#         self.rate = rate
#         super().__init__(layer=keras.layers.Dropout(rate=self.rate))

#     def __str__(self):
#         return f"Dropout({self.rate})"


# class Identity(Operation):
#     def __init__(self):
#         pass

#     def __call__(self, inputs, **kwargs):
#         assert (
#             len(inputs) == 1
#         ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
#         return inputs[0]


# class Activation(Operation):
#     """Activation function operation.

#     Args:
#         activation (callable): an activation function
#     """

#     def __init__(self, activation=None, *args, **kwargs):
#         self.activation = activation
#         self._layer = None

#     def __str__(self):
#         return f"{type(self).__name__}_{self.activation}"

#     def __call__(self, inputs, *args, **kwargs):
#         inpt = inputs[0]
#         if self._layer is None:
#             self._layer = keras.layers.Activation(activation=self.activation)
#         out = self._layer(inpt)
#         return out
