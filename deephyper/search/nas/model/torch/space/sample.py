import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from deephyper.search.nas.model.torch.space.node import VariableNode, ConstantNode
from deephyper.search.nas.model.torch.space.op.op1d import Dense
from deephyper.search.nas.model.torch.space.op.basic import Tensor
from deephyper.search.nas.model.torch.space import TorchSearchSpace


if __name__ == "__main__":
    ss = TorchSearchSpace((4,), (1,))

    in_0 = ConstantNode(op=Tensor(torch.rand(4, 4)), name="in_0")  #! tmp
    ss.input_nodes = [in_0]  #! tmp

    dense = VariableNode("dense")
    dense.add_op(Dense(1, F.relu))

    ss.connect(ss.input_nodes[0], dense)

    ss.set_ops([0])
    model = ss.create_model()

    output = model.forward(x=torch.rand(4, 4))
    print(output)

