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
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
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
    """Run a param_dict on the imdb_bidirectional_lstm benchmark."""
    # Read in values from CLI if no param dict was specified and clean up the param dict.
    param_dict = util.handle_cli(param_dict, build_parser())

    # Display the parsed param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    # Get values from param_dict.
    # Hyperparameters
    BATCH_SIZE     = param_dict["batch_size"]
    DROPOUT        = param_dict["dropout"]
    EMBEDDING_DIMS = param_dict["embedding_dims"]
    EPOCHS         = param_dict["epochs"]
    MAX_FEATURES   = param_dict["max_features"]
    MAXLEN         = param_dict["maxlen"]
    OPTIMIZER      = util.get_optimizer_instance(param_dict)
    UNITS          = param_dict["units"]

    # Other
    model_path    = param_dict["model_path"]

    # Constants
    patience  = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]

    timer.start("stage in")

    # Kept for future implementation of external data loading.
    # if param_dict["data_source"]:
    #     data_source = param_dict["data_source"]
    # else:
    #     data_source = os.path.dirname(os.path.abspath(__file__))
    #     data_source = os.path.join(data_source, "data")

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
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
    if verbose == 1:
        print("x_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

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
        model = Sequential()
        model.add(Embedding(MAX_FEATURES, EMBEDDING_DIMS, input_length=MAXLEN))
        model.add(Bidirectional(LSTM(UNITS)))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    timer.start("model training")

    model.fit(x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=[x_test, y_test], verbose=verbose)

    timer.end() # model training

    score = model.evaluate(x_test, y_test, verbose=verbose)
    acc   = score[1]

    return -acc

def build_parser():
    # Build this benchmark"s cli parser on top of the keras_cli parser.
    parser = build_base_parser()

    # Benchmark specific hyperparameters.
    parser.add_argument("--units", action="store", dest="units",
                        nargs="?", const=1, type=int, default=64,
                        help="units for LSTM")

    parser.add_argument("--max_features", action="store", dest="max_features",
                        nargs="?", const=2, type = int, default=20000,
                        help="max_features when loading data")

    parser.add_argument("--maxlen", action="store", dest="maxlen",
                        nargs="?", const=2, type = int, default=100,
                        help="the max length of the sequence of x_train and x_test")

    parser.add_argument("--embedding_dims", action="store", dest="embedding_dims",
                        nargs="?", const=2, type = int, default = 128,
                        help="how many embedding dims will be added to the model")

    return parser

if __name__ == "__main__":
    run()
