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
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
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


def create_ngram_set(input_list, ngram_value=2):
    
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


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

    ngram_range = 1 
    MAX_FEATURES = param_dict['max_features'] # = 20000
    MAXLEN = param_dict['maxlen'] # = 400
    ENBEDDING_DIMS = param_dict['embedding_dims'] # = 50

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

    timer.end()

    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    ACTIVATION     = util.get_activation_instance(param_dict)
    OPTIMIZER      = util.get_optimizer_instance(param_dict)

    #constants
    patience = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]
    

    timer.start('preprocessing')

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))

        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        start_index = MAX_FEATURES + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        MAX_FEATURES = np.max(list(indice_token.keys())) + 1

        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)



    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)


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

    # if model is None:
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, ENBEDDING_DIMS, input_length=MAXLEN))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation=ACTIVATION))
    model.compile(loss='binary_crossentropy',
                    optimizer=OPTIMIZER,
                    metrics=['accuracy'])

    timer.start('model training')

    model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

    acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)[1]
    timer.end()
    return -acc


def build_parser():
    parser = build_base_parser()

    parser.add_argument('--max_features', action='store', dest='max_features',
                        nargs='?', const=2, type = int, default='20000',
                        help='max_features when loading data')

    parser.add_argument('--maxlen', action='store', dest='maxlen',
                        nargs='?', const=2, type = int, default='400',
                        help='the max length of the sequence of x_train and x_test')
                            
    parser.add_argument('--embedding_dims', action='store', dest='embedding_dims',
                        nargs='?', const=2, type = int, default = '50',
                        help='how many embedding dims will be added to the model')
    return parser


if __name__ == "__main__":
    run()