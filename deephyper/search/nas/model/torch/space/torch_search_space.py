from collections.abc import Iterable
from functools import reduce

import networkx as nx
import torch.nn as nn

from deephyper.core.exceptions.nas.space import (
    InputShapeOfWrongType,
    NodeAlreadyAdded,
    StructureHasACycle,
    WrongOutputShape,
    WrongSequenceToSetOperations,
)

from deephyper.search.nas.model.torch.space import NxSearchSpace


class TorchModel(nn.Module):
    def __init__(self, torch_search_space):
        super().__init__()

        self.ss = torch_search_space

    def forward(self, x):
        return self.ss.forward(x)


class TorchSearchSpace(NxSearchSpace, nn.Module):
    def __init__(self, input_shape, output_shape, *args, **kwargs):

        super().__init__()

        self.input_nodes = None

        self.output_shape = output_shape
        self.output_node = None

        self._model = None

    def set_ops(self, indexes):
        """Set the operations for each node of each cell of the search_space.

        Args:
            indexes (list):  element of list can be float in [0, 1] or int.

        Raises:
            WrongSequenceToSetOperations: raised when 'indexes' is of a wrong length.
        """
        if len(indexes) != len(list(self.variable_nodes)):
            raise WrongSequenceToSetOperations(indexes, list(self.variable_nodes))

        for op_i, node in zip(indexes, self.variable_nodes):
            node.set_op(op_i)

        output_nodes = self.get_output_nodes()

        self.output_node = self.set_output_node(self.graph, output_nodes)

    def set_output_node(self, graph, output_nodes):
        """Set the output node of the search_space.

        Args:
            graph (nx.DiGraph): graph of the search_space.
            output_nodes (Node): nodes of the current search_space without successors.

        Returns:
            Node: output node of the search_space.
        """
        if len(output_nodes) == 1:
            node = output_nodes[0]
        else:
            node = output_nodes
        return node

    def create_model(self):
        """Create the tensors corresponding to the search_space.

        Returns:
            A keras.Model for the current search_space with the corresponding set of operations.
        """
        return TorchModel(self)

    def forward(self, x):
        if type(self.output_node) is list:
            output_tensors = [
                self.create_tensor_aux(self.graph, out) for out in self.output_node
            ]

            for out_T in output_tensors:
                output_n = int(out_T.name.split("/")[0].split("_")[-1])
                out_S = self.output_shape[output_n]
                if out_T.get_shape()[1:] != out_S:
                    raise WrongOutputShape(out_T, out_S)

            input_tensors = [inode._tensor for inode in self.input_nodes]

            return output_tensors
        else:
            output_tensors = self.create_tensor_aux(self.graph, self.output_node)
            if output_tensors.shape[1:] != self.output_shape:
                raise WrongOutputShape(output_tensors, self.output_shape)

            return output_tensors

