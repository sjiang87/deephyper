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
import math
from pprint import pprint
from keras import layers, backend as K
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TerminateOnNaN, EarlyStopping
from deephyper.benchmarks_hps.util import TerminateOnTimeOut
from keras.preprocessing.image import ImageDataGenerator
from deephyper.benchmarks_hps.cliparser import build_base_parser
from deephyper.benchmarks_hps.mnistcnn.loaddata import load_data

# Do some path magic.
here  = os.path.dirname(os.path.abspath(__file__))
top   = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

timer.end() # module loading

def run(param_dict=None, verbose=2):
    """Run a param_dict on the MNISTCNN benchmark."""
    # Read in values from CLI if no param_dict was specified and clean up the param dict.
    param_dict = util.handle_cli(param_dict, build_parser())

    # Display the filled in param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    # Get values from param_dict.
    # Hyperparameters
    ACTIVATION    = util.get_activation_instance(param_dict)
    BATCH_SIZE    = param_dict["batch_size"]
    DROPOUT       = param_dict["dropout"]
    EPOCHS        = param_dict["epochs"]
    F1_SIZE       = param_dict["f1_size"]
    F2_SIZE       = param_dict["f2_size"]
    F1_UNITS      = param_dict["f1_units"]
    F2_UNITS      = param_dict["f2_units"]
    MAX_POOL      = param_dict["max_pool"]
    NUNITS        = param_dict["nunits"]
    OPTIMIZER     = util.get_optimizer_instance(param_dict)
    PADDING_C1    = param_dict["padding_c1"]
    PADDING_C2    = param_dict["padding_c2"]
    PADDING_P1    = param_dict["padding_p1"]
    PADDING_P2    = param_dict["padding_p2"]
    P_SIZE        = param_dict["p_size"]

    # Other
    model_path    = param_dict["model_path"]

    # Constants
    patience  = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]
    num_classes = 10
    # Input image dimensions
    img_rows, img_cols = 28, 28


    timer.start("stage in")

    # Kept for future implementation of external data loading.
    # if param_dict["data_source"]:
    #     data_source = param_dict["data_source"]
    # else:
    #     data_source = os.path.dirname(os.path.abspath(__file__))
    #     data_source = os.path.join(data_source, "data")

    (x_train, y_train), (x_test, y_test) = load_data()

    timer.end() # stage in

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    timer.start("preprocessing")

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    if verbose == 1:
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    if model_path:
        savedModel     = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
        model_mda_path = savedModel.model_mda_path
        model_path     = savedModel.model_path
        model          = savedModel.model
        initial_epoch  = savedModel.initial_epoch
    else:
        model          = None
        model_mda_path = None
        initial_epoch  = 0

    if not model:
        model = Sequential()
        model.add(Conv2D(F1_UNITS, (F1_SIZE, F1_SIZE), padding=PADDING_C1,
                        input_shape=input_shape, activation=ACTIVATION))
        model.add(Conv2D(F1_UNITS, (F1_SIZE, F1_SIZE), padding=PADDING_C1, activation=ACTIVATION))
        if MAX_POOL:
            model.add(MaxPooling2D(pool_size=(P_SIZE, P_SIZE), padding=PADDING_P1))
        model.add(Dropout(DROPOUT))
        model.add(Conv2D(F2_UNITS, (F2_SIZE, F2_SIZE), padding=PADDING_C2, activation=ACTIVATION))
        model.add(Conv2D(F2_UNITS, (F2_SIZE, F2_SIZE), padding=PADDING_C2, activation=ACTIVATION))
        if MAX_POOL:
            model.add(MaxPooling2D(pool_size=(P_SIZE, P_SIZE), padding=PADDING_P2))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(NUNITS, activation=ACTIVATION))
        model.add(Dropout(DROPOUT))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    timer.end() # preprocessing

    timer.start("model training")

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        initial_epoch=initial_epoch,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

    timer.end() # model training

    score = model.evaluate(x_test, y_test, verbose=verbose)
    acc   = score[1]
    if verbose == 1:
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    if model_path:
        timer.start("model save")
        model.save(model_path)
        util.save_meta_data(param_dict, model_mda_path)
        timer.end() # model save

    return -acc

def build_parser():
    # Build this benchmark"s cli parser on top of the base parser.
    parser = build_base_parser()

    # Benchmark specific hyperparameters.
    parser.add_argument("--f1_size", action="store", dest="f1_size",
                        nargs="?", const=2, type=int, default=3,
                        help="Filter 1 dim")

    parser.add_argument("--f2_size", action="store", dest="f2_size",
                        nargs="?", const=2, type=int, default=3,
                        help="Filter 2 dim")

    parser.add_argument("--f1_units", action="store", dest="f1_units",
                        nargs="?", const=2, type=int, default=32,
                        help="Filter 1 units")

    parser.add_argument("--f2_units", action="store", dest="f2_units",
                        nargs="?", const=2, type=int, default=64,
                        help="Filter 2 units")

    parser.add_argument("--max_pool", action="store", dest="max_pool",
                        nargs="?", const=2, type=util.str2bool, default=False, help="include a max pool layer?")

    parser.add_argument("--nunits", action="store", dest="nunits",
                        nargs="?", const=2, type=int, default=512,
                        help="number of units in FC layer")

    parser.add_argument("--padding_c1", action="store", dest="padding_c1",
                        nargs="?", const=2, type=str, default="same",
                        help="padding for the first two convolutional layers; padding can be either \"same\" or \"valid\"")

    parser.add_argument("--padding_c2", action="store", dest="padding_c2",
                        nargs="?", const=2, type=str, default="same",
                        help="padding for the second two convolutional layers; padding can be either \"same\" or \"valid\"")

    parser.add_argument("--padding_p1", action="store", dest="padding_p1",
                        nargs="?", const=2, type=str, default="same",
                        help="padding for the first pooling layer; padding can be either \"same\" or \"valid\"")

    parser.add_argument("--padding_p2", action="store", dest="padding_p2",
                        nargs="?", const=2, type=str, default="same",
                        help="padding for the second pooling layer; padding can be either \"same\" or \"valid\"")

    parser.add_argument("--p_size", action="store", dest="p_size",
                        nargs="?", const=2, type=int, default=2,
                        help="pool size")

    return parser

if __name__ == "__main__":
    run()
