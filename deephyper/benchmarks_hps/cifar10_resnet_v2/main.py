# Set random seeds for reproducability: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
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
import keras
from pprint import pprint
from keras import layers, backend as K
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from hyper2018.benchmarks.util import TerminateOnTimeOut
from keras.preprocessing.image import ImageDataGenerator
from hyper2018.benchmarks.cliparser import build_base_parser
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10


# Do some path magic.
here  = os.path.dirname(os.path.abspath(__file__))
top   = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]


timer.end() # module loading

def run(param_dict={}, verbose=2):
    """Run a param_dict on the MNISTCNN benchmark."""
    # Read in values from CLI if no param dict was specified and clean up the param dict.
    util.handle_cli(param_dict, build_parser())
    
    # Display the filled in param dict.
    if verbose:
        print("PARAM_DICT_CLEAN=")
        pprint(param_dict)

    # Get values from param_dict.
    # Hyperparameters

    # Training parameters
    ACTIVATION          = param_dict['activation']
    BATCH_SIZE          = param_dict["batch_size"] #=32
    BASE_LR             = param_dict['base_lr']
    DATA_AUG            = param_dict['data_augmentation']
    EPOCHS              = param_dict['epochs']
    KERNEL_SIZE         = param_dict['kernel_size']
    LR80                = param_dict['lr80']
    LR120               = param_dict['lr120']
    LR160               = param_dict['lr160']
    LR180               = param_dict['lr180']
    NUM_FILTERS         = param_dict['num_filters']
    NUM_FILTERS_IN      = param_dict["num_filters_in"] #16
    OPTIMIZER           = util.get_optimizer_instance(param_dict)


    #param_dict["n"] is used only when testing different depths of the resnet

    # Other
    LOSS_FUNCTION = param_dict["loss_function"]
    METRICS       = param_dict["metrics"]
    MODEL_PATH    = param_dict["model_path"]

    # Constants
    num_classes = 10
    subtract_pixel_mean = True
    N = 3
    depth = N * 9 + 2
    model_type = 'ResNet%dv%d' % (depth, 2)

    timer.start("stage in")

    if param_dict["data_source"]:
        data_source = param_dict["data_source"]
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, "data")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    timer.end() # stage in

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    

    timer.start("preprocessing")

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    def lr_schedule(epoch):

        lr = BASE_LR #1e-3
        if epoch > 180:
            lr *= LR180 #0.5e-3
        elif epoch > 160:
            lr *= LR160 #1e-3
        elif epoch > 120:
            lr *= LR120# 1e-2
        elif epoch > 80:
            lr *= LR80 #1e-1

        return lr
    
    def resnet_layer(inputs,
                 num_filters=NUM_FILTERS,
                 kernel_size=KERNEL_SIZE,
                 strides=1,
                 activation=ACTIVATION,
                 batch_normalization=True,
                 conv_first=True):
        
        conv = Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(ACTIVATION)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(ACTIVATION)(x)
            x = conv(x)
        return x




    def resnet_v2(input_shape, depth, num_classes=10):
    
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = NUM_FILTERS_IN
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=inputs,
                        num_filters=num_filters_in,
                        conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = resnet_layer(inputs=x,
                                num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)
                y = resnet_layer(inputs=y,
                                num_filters=num_filters_in,
                                conv_first=False)
                y = resnet_layer(inputs=y,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

            # Add classifier on top.
            # v2 has BN-ReLU before Pooling
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model

    if MODEL_PATH:
        savedModel     = util.resume_from_disk(BNAME, param_dict, data_dir=MODEL_PATH)
        model_mda_path = savedModel.model_mda_path
        MODEL_PATH     = savedModel.model_path
        model          = savedModel.model
        initial_epoch  = savedModel.initial_epoch
    else:
        model          = None
        model_mda_path = None
        initial_epoch  = 0

    if model is None:
        model = resnet_v2(input_shape=input_shape, depth=depth)
        model.compile(loss='categorical_crossentropy',
                    optimizer=OPTIMIZER,
                    metrics=['accuracy'])
        #model.summary()

    timer.end() # preprocessing

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)


    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, TerminateOnNaN]

    # Run training, with or without data augmentation.
    if not DATA_AUG:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

    timer.start("model training")

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(x_test, y_test),
                        epochs=EPOCHS, verbose=1, workers=4,
                        callbacks=callbacks)

    score             = model.evaluate(x_test, y_test, verbose=verbose)

    timer.end() # model training

    
    negative_accuracy = -score[1]

    if MODEL_PATH:
        timer.start("model save")
        model.save(MODEL_PATH)
        util.save_meta_data(param_dict, model_mda_path)
        timer.end() # model save

    return negative_accuracy

def build_parser():
    # Build this benchmark's cli parser on top of the base parser.
    parser = build_base_parser()

    # Benchmark specific hyperparameters.
    parser.add_argument("--base_lr", action="store", dest="base_lr",
                        nargs="?", const=2, type=int, default=1e-3,)

    parser.add_argument("--lr80", action="store", dest="lr80",
                        nargs="?", const=2, type=int, default=1e-1)

    parser.add_argument("--lr120", action="store", dest="lr120",
                        nargs="?", const=2, type=int, default=1e-2)

    parser.add_argument("--lr160", action="store", dest="lr160",
                        nargs="?", const=2, type=int, default=1e-3)

    parser.add_argument("--lr180", action="store", dest="lr180",
                        nargs="?", const=2, type=int, default=0.5e-3)

    parser.add_argument("--num_filters", action="store", dest="num_filters",
                        nargs="?", const=2, type=int, default=16)

    parser.add_argument("--num_filters_in", action="store", dest="num_filters_in",
                        nargs="?", const=2, type=str, default=16)

    # Model evaluation metrics.
    parser.add_argument("--loss_function", action="store", dest="loss_function",
                        nargs="?", const=2, type=str, default="categorical_crossentropy",
                        help="the loss function to use when reporting evaluation information.")

    parser.add_argument("--metrics", action="store", dest="metrics",
                        nargs="?", const=2, type=str, default="accuracy",
                        help="the metric used when compiling and evaluating the benchmark")

    return parser

if __name__ == "__main__":
    run()
