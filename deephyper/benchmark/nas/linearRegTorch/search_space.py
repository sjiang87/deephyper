import torch
import torch.nn as nn
import torch.nn.functional as F

from deephyper.search.nas.model.torch.space.torch_search_space import TorchSearchSpace
from deephyper.search.nas.model.torch.space.op.op1d import Dense
from deephyper.search.nas.model.torch.space.node import VariableNode


def gen_vnode():
    vnode1 = VariableNode()
    for n in range(1, 11):
        vnode1.add_op(Dense(n, F.relu))
    return vnode1


def create_search_space(input_shape=(2,), output_shape=(1,), **kwargs):
    ss = TorchSearchSpace(input_shape, output_shape)

    vnode = gen_vnode()
    ss.connect(ss.input_nodes[0], vnode)

    # if type(input_shape) is list:
    #     vnodes = []
    #     for i in range(len(input_shape)):
    #         vn = gen_vnode()
    #         vnodes.append(vn)
    #         ss.connect(ss.input_nodes[i], vn)

    #     cn = ConstantNode()
    #     cn.set_op(Concatenate(ss, vnodes))

    #     vn = gen_vnode()
    #     ss.connect(cn,vn)

    # else:
    #     vnode1 = gen_vnode()
    #     ss.connect(ss.input_nodes[0], vnode1)

    # return ss


if __name__ == "__main__":
    create_search_space()

