# Set random seeds for reproducability: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from deephyper.benchmarks_hps import util
timer = util.Timer()
timer.start("module loading")

import sys
import keras
from pprint import pprint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing import sequence
from deephyper.benchmarks_hps.cliparser import build_base_parser
from keras.callbacks import EarlyStopping, TerminateOnNaN
import math

# Do some path magic.
here  = os.path.dirname(os.path.abspath(__file__))
top   = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

timer.end() # module loading

def run(param_dict=None, verbose=2):
    """Run a param_dict on the imdbcnn benchmark."""
    # Read in values from CLI if no param dict was specified and clean up the param dict.
    param_dict = util.handle_cli(param_dict, build_parser())

    # Display the filled in param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    timer.start("stage in")

    # if param_dict['data_source']:
    #     data_source = param_dict['data_source']
    # else:
    #     data_source = os.path.dirname(os.path.abspath(__file__))
    #     data_source = os.path.join(data_source, 'data')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    timer.end()



    #hyperparameters
    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    DROPOUT = param_dict['dropout']
    ACTIVATION = util.get_activation_instance(param_dict['activation'], param_dict['alpha'])
    NHIDDEN = param_dict['nhidden']
    NUNITS = param_dict['nunits']
    OPTIMIZER      = util.get_optimizer_instance(param_dict)
    
    #constants
    num_classes = 10
    patience  = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]

    timer.start('preprocessing')

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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

    # if model is None:
    model = Sequential()
    model.add(Dense(NUNITS, activation=ACTIVATION, input_shape=(784,)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
            optimizer=OPTIMIZER,
            metrics=['accuracy'])

    timer.end()

    timer.start('model training')
    history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    #initial_epoch=initial_epoch,
                    epochs=EPOCHS,
                    #verbose=1,
                    callbacks=callbacks,
                    #validation_split = 0.3)
                    validation_data=(x_test, y_test))
    timer.end()
   
    
    # if model_path:
    #     timer.start('model save')
    #     model.save(model_path)  
    #     util.save_meta_data(param_dict, model_mda_path)
    #     timer.end()

    score = model.evaluate(x_test, y_test, verbose=verbose)
    acc   = score[1]

    return -acc


def build_parser():
    # Build this benchmark"s cli parser on top of the keras_cli parser.
    parser = build_base_parser()

    # Benchmark specific hyperparameters.

    parser.add_argument('--nunits', action='store', dest='nunits',
                        nargs='?', const=2, type=int, default='512',
                        help='number of units/layer in MLP')

    parser.add_argument('--nhidden', action='store', dest='nhidden',
                        nargs='?', const=2, type=int, default='2',
                        help='number of hidden layers in MLP')

    
    return parser

if __name__ == "__main__":
    run()