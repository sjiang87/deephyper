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

import math
from deephyper.benchmarks_hps.cliparser import build_base_parser
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import TerminateOnNaN, EarlyStopping

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

    #print('Loading data...')

    MAX_FEATURES = param_dict['max_features'] #20000

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

    timer.end()

    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    DROPOUT = param_dict['dropout']
    ACTIVATION1  = util.get_activation_instance(param_dict['activation1'], param_dict['alpha1'])
    ACTIVATION2  = util.get_activation_instance(param_dict['activation2'], param_dict['alpha2'])
    PADDING = param_dict["padding"]
    METRICS = param_dict['metrics']
    LOSS_FUNCTION = param_dict['loss_function']
    FILTERS = param_dict['filters']
    POOL_SIZE = param_dict['pool_size']
    STRIDE = param_dict['stride']
    OPTIMIZER = util.get_optimizer_instance(param_dict)
    
    MAXLEN = param_dict['maxlen'] #100
    EMBEDDING_SIZE = param_dict['embedding_size'] #128
    LSTM_OUTPUT_SIZE = param_dict['lstm_output_size']  #70
    KERNEL_SIZE = param_dict['kernel_size'] #5

    #constants
    patience = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]




    #print(len(x_train), 'train sequences')
    #print(len(x_test), 'test sequences')

    timer.start('preprocessing')

    #print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
    #print('x_train shape:', x_train.shape)
    #print('x_test shape:', x_test.shape)

    timer.end()

    # model_path = param_dict['model_path']
    # model_mda_path = None
    # model = None
    # initial_epoch = 0

    # if model_path:
    #     savedModel = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
    #     model_mda_path = savedModel.model_mda_path
    #     model_path = savedModel.model_path
    #     model = savedModel.model
    #     initial_epoch = savedModel.initial_epoch

    #if model is None:
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, EMBEDDING_SIZE, input_length=MAXLEN))
    model.add(Dropout(DROPOUT))
    model.add(Conv1D(FILTERS,
                    KERNEL_SIZE,
                    padding=PADDING,
                    activation=ACTIVATION1,
                    strides=STRIDE))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(LSTM(LSTM_OUTPUT_SIZE))
    model.add(Dense(1))
    model.add(Activation(ACTIVATION2))

    model.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=[METRICS])

    timer.start('model training')

    model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_test, y_test),
          callbacks=callbacks)
    score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    # print('Test score:', score)
    # print('Test accuracy:', acc)

    timer.end()
    return -acc


def build_parser():
    # Build this benchmark"s cli parser on top of the base parser.
    parser = build_base_parser()

    parser.add_argument('--padding', action='store', dest='padding',
                            nargs='?', const=2, default='same',
                            help='padding can be either "same" or "none"')

    parser.add_argument('--metrics', action='store', dest='metrics',
                        nargs='?', const=2, default='accuracy',
                        help='the metric used when compiling and evaluating the benchmark')
    
    parser.add_argument('--pool_size', action='store', dest='pool_size',
                        nargs='?', const=2, type = int, default='2',
                        )
                        
    parser.add_argument('--max_features', action='store', dest='max_features',
                        nargs='?', const=2, type= int, default='20000',
                        )

    parser.add_argument('--maxlen', action='store', dest='maxlen',
                        nargs='?', const=2, type = int, default='100',
                        )

    parser.add_argument('--embedding_size', action='store', dest='embedding_size',
                        nargs='?', const=2, type = int, default='128',
                        )

    parser.add_argument('--lstm_output_size', action='store', dest='lstm_output_size',
                        nargs='?', const=2, type = int, default='70',
                        )

    parser.add_argument('--kernel_size', action='store', dest='kernel_size',
                        nargs='?', const=2, type = int, default='5',
                        )

    parser.add_argument('--filters', action='store', dest='filters', nargs='?', 
                        const=2, type=int, default='64', help='number of filters')

    parser.add_argument('--stride', action='store', dest='stride', nargs='?', 
                        const=2, type=int, default='1', help='number of strides')
    return parser


if __name__ == "__main__":
    run()