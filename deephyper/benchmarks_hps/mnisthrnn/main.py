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




import keras
import math
import sys
from pprint import pprint
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.callbacks import TerminateOnNaN, EarlyStopping
from hyper2018.benchmarks.util import TerminateOnTimeOut
from keras.preprocessing.image import ImageDataGenerator
from hyper2018.benchmarks.cliparser import build_base_parser
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
from hyper2018.benchmarks.util import TerminateOnTimeOut
from keras.models import load_model


timer.end()
here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]



def run(param_dict=None, verbose=2):
    """Run a param_dict on the MNISTCNN benchmark."""
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

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    timer.end()

    # input image dimensions

    row_hidden = 128
    col_hidden = 128


    img_rows, img_cols = 28, 28



    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, (img_rows), (img_cols))
        x_test = x_test.reshape(x_test.shape[0], 1, (img_rows), (img_cols))

    else:
        x_train = x_train.reshape(x_train.shape[0], (img_rows), (img_cols), 1)
        x_test = x_test.reshape(x_test.shape[0], (img_rows), (img_cols), 1)


    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    OPTIMIZER     = util.get_optimizer_instance(param_dict)
    
    #others

    #model_path = ''

    # Constants
    patience  = math.ceil(EPOCHS/2)
    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=patience, verbose=verbose, mode="auto"),
        TerminateOnNaN()]
    num_classes = 10

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if verbose ==1:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    timer.start('preprocessing')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    row, col, pixel = x_train.shape[1:]

    # 4D input.
    x = Input(shape=(row, col, pixel))

    

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
    #     # Encodes a row of pixels using TimeDistributed Wrapper.
    #     encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    #     # Encodes columns of encoded rows.
    #     encoded_columns = LSTM(col_hidden)(encoded_rows)
    #     #final prediction
    #     prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    #     model = Model(x, prediction)
    #     model.compile(loss=LOSS_FUNCTION,
    #                 optimizer=optimizer,
    #                 metrics=[METRICS])
# Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    # Encodes columns of encoded rows.
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    #final prediction
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model.compile(loss='categorical_crossentropy',
                optimizer=OPTIMIZER,
                metrics=['accuracy'])


    timer.end()


    timer.start('model training')

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        #initial_epoch=initial_epoch,
                        verbose=1, 
                        callbacks=callbacks,
                        #validation_split = 0.3,
                        validation_data = (x_test, y_test))
    
    timer.end()

    score = model.evaluate(x_test, y_test, verbose=0)
    
    if verbose ==1:
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
       
    # if model_path:
    #     timer.start('model save')
    #     model.save(model_path)  
    #     util.save_meta_data(param_dict, model_mda_path)
    #     timer.end()

    #print('OUTPUT:', -score[1])
    return -score[1]

def build_parser():
    parser = build_base_parser()
    return parser

if __name__ == "__main__":
    run()
