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

from hyper2018.benchmarks import util
timer = util.Timer()
timer.start("module loading")

import sys
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing import sequence
from hyper2018.benchmarks.cliparser import build_base_parser
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

    # Kept for future implementation of external data loading.
    # if param_dict["data_source"]:
    #     data_source = param_dict["data_source"]
    # else:
    #     data_source = os.path.dirname(os.path.abspath(__file__))
    #     data_source = os.path.join(data_source, "data")

    # Get arguments from param_dict.
    # Hyperparameters
    ACTIVATION     = util.get_activation_instance(param_dict)
    BATCH_SIZE     = param_dict["batch_size"]
    DROPOUT        = param_dict["dropout"]
    EMBEDDING_DIMS = param_dict["embedding_dims"]
    EPOCHS         = param_dict["epochs"]
    FILTERS        = param_dict["filters"]
    HIDDEN_DIMS    = param_dict["hidden_dims"]
    KERNEL_SIZE    = param_dict["kernel_size"]
    MAX_FEATURES   = param_dict["max_features"]
    MAXLEN         = param_dict["maxlen"]
    OPTIMIZER      = util.get_optimizer_instance(param_dict)
    PADDING        = param_dict["padding"]
    STRIDES        = param_dict["strides"]

    # Other
    model_path     = param_dict["model_path"]

    # Constants
    patience  = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]

    # Load data.
    if verbose == 1:
        print("Loading data...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    if verbose == 1:
        print(len(x_train), "train sequences")
        print(len(x_test), "test sequences")

    timer.end() # stage in

    timer.start("preprocessing")

    # Pad sequences (samples x time)
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test  = sequence.pad_sequences(x_test, maxlen=MAXLEN)

    timer.end() # preprocessing

    if model_path:
        savedModel     = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
        model_mda_path = savedModel.model_mda_path
        model_path     = savedModel.model_path
        model          = savedModel.model
        initial_epoch  = savedModel.initial_epoch
    else:
        model_mda_path = None
        model          = None
        initial_epoch  = 0

    if not model:
        if verbose == 1:
            print("Build model...")

        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(MAX_FEATURES,
                            EMBEDDING_DIMS,
                            input_length=MAXLEN))
        model.add(Dropout(DROPOUT))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(FILTERS,
                        KERNEL_SIZE,
                        padding=PADDING,
                        activation=ACTIVATION,
                        strides=STRIDES))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(HIDDEN_DIMS))
        model.add(Dropout(DROPOUT))
        model.add(Activation(ACTIVATION))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                    optimizer=OPTIMIZER,
                    metrics=["accuracy"])

    timer.start("model training")

    model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=callbacks,
          validation_data=(x_test, y_test), verbose=verbose)

    timer.end() # model training

    score = model.evaluate(x_test, y_test, verbose=verbose)
    acc   = score[1]

    return -acc

def build_parser():
    # Build this benchmark"s cli parser on top of the keras_cli parser.
    parser = build_base_parser()

    # Benchmark specific hyperparameters.
    parser.add_argument("--embedding_dims", action="store", dest="embedding_dims",
                        nargs="?", const=2, type = int, default = 50,
                        help="how many embedding dims will be added to the model")

    parser.add_argument("--filters", action="store", dest="filters",
                        nargs="?", const=2, type = int, default = 250,
                        help="the number of output filters in the convolution")

    parser.add_argument("--hidden_dims", action="store", dest="hidden_dims",
                        nargs="?", const=2, type = int, default = 250,
                        help="hidden dims of a vanilla hidden layer")

    parser.add_argument("--kernel_size", action="store", dest="kernel_size",
                        nargs="?", const=2, type = int, default = 3,
                        help="the length of the 1D convolution window")

    parser.add_argument("--max_features", action="store", dest="max_features",
                        nargs="?", const=2, type = int, default=5000,
                        help="max_features when loading data")

    parser.add_argument("--maxlen", action="store", dest="maxlen",
                        nargs="?", const=2, type = int, default=400,
                        help="the max length of the sequence of x_train and x_test")

    parser.add_argument("--padding", action="store", dest="padding",
                        nargs="?", const=2, type=str, default="valid",
                        help="padding can be either \"same\" or \"valid\"")

    parser.add_argument("--strides", action="store", dest="strides", nargs="?", const=2, type=int, default=1, help="number of strides")

    # Model evaluation metrics.
    parser.add_argument("--loss_function", action="store", dest="loss_function",
                        nargs="?", const=2, type=str, default="binary_crossentropy",
                        help="the loss function to use when reporting evaluation information.")

    parser.add_argument("--metrics", action="store", dest="metrics",
                        nargs="?", const=2, type=str, default="accuracy",
                        help="the metric used when compiling and evaluating the benchmark")

    return parser

if __name__ == "__main__":
    run()
