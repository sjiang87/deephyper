# Set random seeds for reproducability: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# from keras import backend as K
tf.set_random_seed(1234)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

from deephyper.benchmarks_hps import util
timer = util.Timer()
timer.start("module loading")

import sys
import math
import keras.utils
from pprint import pprint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.datasets import reuters
from keras.callbacks import TerminateOnNaN, EarlyStopping
from keras.preprocessing.text import Tokenizer
from deephyper.benchmarks_hps.cliparser import build_base_parser

# Do some path magic.
here  = os.path.dirname(os.path.abspath(__file__))
top   = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

timer.end() # module loading

def run(param_dict=None, verbose=2):
    """Run a param_dict on the reutersmlp benchmark."""
    # Read in values from CLI if no param dict was specified and clean up the param dict.
    param_dict = util.handle_cli(param_dict, build_parser())

    # Display the parsed param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    # Get values from param_dict.
    # Hyperparameters
    ACTIVATION    = util.get_activation_instance(param_dict)
    BATCH_SIZE    = param_dict["batch_size"]
    DROPOUT       = param_dict["dropout"]
    EPOCHS        = param_dict["epochs"]
    MAX_WORDS     = param_dict["max_words"]
    NUNITS        = param_dict["nunits"]
    OPTIMIZER     = util.get_optimizer_instance(param_dict)
    SKIP_TOP      = param_dict["skip_top"]

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

    # Load data.
    if verbose == 1:
        print("Loading data...")
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=MAX_WORDS, skip_top = SKIP_TOP, test_split=0.2)

    if verbose == 1:
        print(len(x_train), "train sequences")
        print(len(x_test), "test sequences")

    timer.end() # stage in

    timer.start("preprocessing")

    num_classes = np.max(y_train) + 1
    if verbose == 1:
        print(num_classes, "classes")

    # Vectorize sequence data.
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    x_train   = tokenizer.sequences_to_matrix(x_train, mode="binary")
    x_test    = tokenizer.sequences_to_matrix(x_test, mode="binary")
    # Convert class vector to binary class matrix for use with categorical_crossentropy.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    if verbose == 1:
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

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
            print("Building model...")
        model = Sequential()
        model.add(Dense(NUNITS, input_shape=(MAX_WORDS,), activation=ACTIVATION))
        model.add(Dropout(DROPOUT))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                    optimizer=OPTIMIZER,
                    metrics=["accuracy"])

    timer.end() # preprocessing

    timer.start("model training")

    history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=verbose,
                    validation_split=0.1,
                    callbacks=callbacks)

    timer.end() # model training

    score = model.evaluate(x_test, y_test,
                    batch_size=BATCH_SIZE, verbose=verbose)
    acc   = score[1]
    if verbose == 1:
        print("Test score:", score[0])
        print("Test accuracy:", score[1])

    return -acc

def build_parser():
    # Build this benchmark"s cli parser on top of the keras_cli parser.
    parser = build_base_parser()

    # Benchmark specific hyperparameters.
    parser.add_argument("--nunits", action="store", dest="nunits",
                        nargs="?", const=1, type=int, default=512,
                        help="Dense units")

    parser.add_argument("--max_words", action="store", dest="max_words",
                        nargs="?", const=2, type=int, default=1000)

    parser.add_argument("--skip_top", action="store", dest="skip_top",
                        nargs="?", const=2, type=int, default=0)

    return parser

if __name__ == "__main__":
    run()
