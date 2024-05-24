from collections import OrderedDict

import tensorflow as tf

optimizers_keras = OrderedDict()
optimizers_keras["sgd"] = tf.keras.optimizers.legacy.SGD
optimizers_keras["rmsprop"] = tf.keras.optimizers.legacy.RMSprop
optimizers_keras["adagrad"] = tf.keras.optimizers.legacy.Adagrad
optimizers_keras["adam"] = tf.keras.optimizers.legacy.Adam
optimizers_keras["adadelta"] = tf.keras.optimizers.legacy.Adadelta
optimizers_keras["adamax"] = tf.keras.optimizers.legacy.Adamax
optimizers_keras["nadam"] = tf.keras.optimizers.legacy.Nadam


def selectOptimizer_keras(name):
    """Return the optimizer defined by name."""
    if optimizers_keras.get(name) is None:
        raise RuntimeError('"{0}" is not a defined optimizer for keras.'.format(name))
    else:
        return optimizers_keras[name]


def check_data_config(data_dict):
    gen_keys = ["train_gen", "train_size", "valid_gen", "valid_size", "types", "shapes"]
    ndarray_keys = ["train_X", "train_Y", "valid_X", "valid_Y"]
    if all([k in data_dict.keys() for k in gen_keys]):
        return "gen"
    elif all([k in data_dict.keys() for k in ndarray_keys]):
        return "ndarray"
    else:
        raise RuntimeError("Wrong data config...")
