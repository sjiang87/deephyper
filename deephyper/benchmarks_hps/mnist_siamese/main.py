from __future__ import print_function
from __future__ import absolute_import
import sys
from pprint import pprint
import os
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from deephyper.benchmarks_hps import util 

timer = util.Timer()
timer.start('module loading')

import random
import keras
import math
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from deephyper.benchmarks_hps.cliparser import build_base_parser
from deephyper.benchmarks_hps.util import TerminateOnTimeOut
from keras.callbacks import EarlyStopping, TerminateOnNaN

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]


timer.end()


def run(param_dict=None, verbose=2):
    # Read in values from CLI if no param_dict was specified and clean up the param dict.
    param_dict = util.handle_cli(param_dict, build_parser())

    # Display the filled in param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    timer.start('stage in')

    if param_dict['data_source']:
        data_source = param_dict['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data')

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        MARGIN = int(param_dict['margin']) # = 1
        return K.mean(y_true * K.square(y_pred) +
                    (1 - y_true) * K.square(K.maximum(MARGIN - y_pred, 0)))


    def create_pairs(x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)


    def create_base_network(input_shape):
        '''Base network to be shared (eq. to feature extraction).
        '''
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(UNITS, activation=ACTIVATION)(x)
        x = Dropout(DROPOUT)(x)
        x = Dense(UNITS, activation=ACTIVATION)(x)
        x = Dropout(DROPOUT)(x)
        x = Dense(UNITS, activation=ACTIVATION)(x)
        return Model(input, x)


    def compute_accuracy(y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)


    def accuracy(y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    timer.end()

    num_classes = 10
    BATCH_SIZE      = param_dict['batch_size']
    EPOCHS          = param_dict['epochs']
    DROPOUT         = param_dict['dropout']
    ACTIVATION      = util.get_activation_instance(param_dict)
    UNITS           = param_dict['units']
    OPTIMIZER       = util.get_optimizer_instance(param_dict)
    patience  = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]

    timer.start('preprocessing')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    input_shape = x_train.shape[1:]

    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    timer.end()

    
    #Possible future impleentation: saving model path
    #                               and start from initial_epoch

    #model_path = param_dict['model_path']
    #model_mda_path = None
    #model = None
    #initial_epoch = 0

    # if model_path:
    #     savedModel = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
    #     model_mda_path = savedModel.model_mda_path
    #     model_path = savedModel.model_path
    #     model = savedModel.model
    #     initial_epoch = savedModel.initial_epoch

    # if model is None:
    #     base_network = create_base_network(input_shape)

    #     input_a = Input(shape=input_shape)
    #     input_b = Input(shape=input_shape)

    #     processed_a = base_network(input_a)
    #     processed_b = base_network(input_b)

    #     distance = Lambda(euclidean_distance,
    #                     output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    #     model = Model([input_a, input_b], distance)
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    model.compile(loss=contrastive_loss, optimizer=OPTIMIZER, metrics=[accuracy])

    timer.end()





    timer.start('model training')

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          callbacks = callbacks)

    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    timer.end()
    return -te_acc


def build_parser():
    # Build this benchmark"s cli parser on top of the base parser.
    parser = build_base_parser()
    parser.add_argument('--units', action='store', dest='units',
                        nargs='?', const=2, type=int, default='128',
                        help='dimensionality of the output space')

    parser.add_argument('--margin', action='store', dest='margin',
                        nargs='?', const=2, type=int, default='1',
                        help='margin defines the radius around G_W(X)')
    
    return parser


if __name__ == "__main__":
    run()