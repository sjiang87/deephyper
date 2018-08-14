import os
import time
import numpy as np
import pickle
import hashlib
from pprint import pprint
from datetime import datetime
from filelock import FileLock
from tensorflow import set_random_seed as tf_seed
from collections import namedtuple
from keras.callbacks import Callback
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.activations import softmax, elu, relu, softplus, softsign, selu, tanh, sigmoid, hard_sigmoid, linear

def handle_cli(initial_param_dict, parser):
    """Handle CLI and param_dict cleanup for the benchmarks' main functions."""
    # If an initial_param_dict was specified, create a copy of it so any
    # reference to the intial_param_dict is not tarnished.
    # If no initial_param_dict was specified, read arguments in from cli.
    if initial_param_dict:
        param_dict = {}
        param_dict.update(initial_param_dict)
        # Fill in missing arguments from initial_param_dict.
        fill_missing_defaults(param_dict, parser)
        # Cast all values in param_dict that have default values to their default type.
        match_parser_types(param_dict, parser)
    else:
        cli_args   = parser.parse_args()
        param_dict = vars(cli_args)

    return param_dict

def expect(_dict, key, default_value):
    """Ensure that a value for 'key' is specified in the dictionary.
    If a value is not already specified for 'key', set its value to 'default_value'.
    Return the value that 'key' now has in the dictionary."""
    if key in _dict:
        return _dict[key]
    else:
        _dict[key] = default_value
        return default_value

def fill_missing_defaults(_dict, parser):
    """Return _dict filled in
    with missing values that were not supplied directly in CLI."""
    default_params = vars(parser.parse_args(""))

    missing = (k for k in default_params if k not in _dict)
    for k in missing:
        _dict[k] = default_params[k]

    return

def match_parser_types(_dict, parser):
    """Given a dictionary, cast all its values to the default types of the default values in the argument parser."""
    default_dict = vars(parser.parse_args(""))
    for key, default_value in default_dict.items():
        if key in _dict:
            value        = _dict[key]
            value_type   = type(value)
            default_type = type(default_value)
            if value_type == str and default_type == bool:
                _dict[key] = str2bool(value)
            elif not value_type == default_type:
                _dict[key] = default_type(value)
    return

def get_optimizer_instance(param_dict):
    """Construct the appropriate optimizer from the param_dict
    and return `None` if an optimizer instance is unable to be instantiated."""
    # Get relevant optimizer parameters from the param_dict.
    optimizer_str = param_dict["optimizer"]
    learning_rate = param_dict["learning_rate"]
    momentum      = param_dict["momentum"]
    decay         = param_dict["decay"]
    nesterov      = param_dict["nesterov"]
    rho           = param_dict["rho"]
    epsilon       = param_dict["epsilon"]
    beta_1        = param_dict["beta_1"]
    beta_2        = param_dict["beta_2"]
    amsgrad       = param_dict["amsgrad"]

    # Determine the optimizer constructor based on optimizer_str
    # and pass in appropriate values.
    if optimizer_str == "sgd":
        optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)
    elif optimizer_str == "rmsprop":
        optimizer = RMSprop(lr=learning_rate, rho=rho, epsilon=epsilon, decay=decay)
    elif optimizer_str == "adagrad":
        optimizer = Adagrad(lr=learning_rate, epsilon=epsilon, decay=decay)
    elif optimizer_str == "adadelta":
        optimizer = Adadelta(lr=learning_rate, rho=rho, epsilon=epsilon, decay=decay)
    elif optimizer_str == "adam":
        optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    elif optimizer_str == "adamax":
        optimizer = Adamax(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    elif optimizer_str == "nadam":
        optimizer = Nadam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    else:
        optimizer = None

    return optimizer

def get_activation_instance(activation_str, alpha=None):
    """Construct the appropriate activation function from the param_dict
    and return `None` if an activation function instance is unable to be instantiated."""

    if activation_str == "elu":
        if alpha is None:
            alpha = 1.0
        activation = lambda x: elu(x, alpha)
    elif activation_str == "softplus":
        activation = softplus
    elif activation_str == "softsign":
        activation = softsign
    elif activation_str == "relu":
        if alpha is None:
            alpha = 0.0
        activation = lambda x: relu(x, alpha)
    elif activation_str == "tanh":
        activation = tanh
    elif activation_str == "sigmoid":
        activation = sigmoid
    elif activation_str == "hard_sigmoid":
        activation = hard_sigmoid
    elif activation_str == "linear":
        activation = linear
    else:
        activation = None

    return activation

def str2bool(s):
    s = s.lower().strip()
    if s == 'false':
        return False
    else:
        return True

class Timer:
    def __init__(self):
        self.t0 = 0.0
        self.name = None
    def start(self, name):
        self.name = name
        self.t0 = time.time()
    def end(self):
        elapsed = time.time() - self.t0
        if not self.name: return
        print("TIMER " + str(self.name) + ": " + str(np.around(elapsed, 4)) + " seconds")
        self.t0 = 0.0
        self.name = None

def extension_from_parameters(param_dict):
    EXCLUDE_PARAMS = ['epochs', 'model_path', 'data_source',
                      'stage_in_destination', 'version',
                      'backend']
    extension = ''
    for key in sorted(param_dict):
        if key not in EXCLUDE_PARAMS:
            extension += '.{}={}'.format(key,param_dict[key])
    print("extension:", extension)
    return extension

def save_meta_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_meta_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def resume_from_disk(benchmark_name, param_dict, data_dir='', custom_objects={}):
    from keras.models import load_model
    SavedModel = namedtuple('SavedModel', ['model', 'model_path',
                                       'initial_epoch', 'model_mda_path']
                            )
    extension = extension_from_parameters(param_dict)
    hex_name = hashlib.sha224(extension.encode('utf-8')).hexdigest()
    model_name = '{}-{}.h5'.format(benchmark_name, hex_name)
    model_mda_name = '{}-{}.pkl'.format(benchmark_name, hex_name)

    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    model_path = os.path.join(data_dir, model_name)
    model_mda_path = os.path.join(data_dir, model_mda_name)

    initial_epoch = 0
    model = None

    if os.path.exists(model_path) and os.path.exists(model_mda_path):
        print('model and meta data exists; loading model from h5 file')
        if benchmark_name == 'regression':
            with open(model_path, 'rb') as fp: model = pickle.load(fp)
        else:
            model = load_model(model_path, custom_objects=custom_objects)

        saved_param_dict = load_meta_data(model_mda_path)
        initial_epoch = saved_param_dict['epochs']
        if initial_epoch < param_dict['epochs']:
            print("loading from epoch " + str(initial_epoch))
            print("running to epoch " + str(param_dict['epochs']))
        else:
            raise RuntimeError("Specified Epochs is less than the initial epoch; will not run")
    else:
        print("Did not find saved model at", model_path)

    return SavedModel(model=model, model_path=model_path,
              model_mda_path=model_mda_path,
              initial_epoch=initial_epoch)

def stage_in(file_names, source, dest):
    from keras.utils.data_utils import get_file
    print("Stage in files:", file_names)
    print("From source dir:", source)
    print("To destination:", dest)

    paths = {}
    for name in file_names:
        origin = os.path.join(source, name)
        assert os.path.exists(origin), str(origin) + ' not found'

        if os.path.exists(dest):
            target = os.path.join(dest, name)
            with FileLock(target + '.lock'):
                paths[name] = get_file(fname=target, origin='file://'+origin)
        else:
            paths[name] = origin

        print("File " + str(name) + " will be read from " + str(paths[name]))
    return paths

class TerminateOnTimeOut(Callback):
    def __init__(self, timeout_in_min = 10):
        super(TerminateOnTimeOut, self).__init__()
        self.run_timestamp = None
        self.timeout_in_sec = timeout_in_min * 60
        #self.validation_data = validation_data
    def on_train_begin(self, logs={}):
        self.run_timestamp = datetime.now()
    def on_batch_end(self, epoch, logs={}):
        run_end = datetime.now()
        run_duration = run_end - self.run_timestamp
        run_in_sec = run_duration.total_seconds() #/ (60 * 60)
        #print(' - current training time = %2.3fs/%2.3fs' % (run_in_sec, self.timeout_in_sec))
        if self.timeout_in_sec != -1:
            if run_in_sec >= self.timeout_in_sec:
                print(' - timeout: training time = %2.3fs/%2.3fs' % (run_in_sec, self.timeout_in_sec))
                #print('TimeoutRuntime: %2.3fs, Maxtime: %2.3fs' % (run_in_sec, self.timeout_in_sec))
                self.model.stop_training = True
                #if self.validation_data is not None:
                #    x, y = self.validation_data[0], self.validation_data[1]
                #    loss, acc = self.model.evaluate(x,y)
                #    #print(self.model.history.keys())

def test_input(param_dict, optimizer):
    """A partial sanity check on cleaned up paramparam_dict."""
    # Ensure optimizer was instantiated correctly.
    if param_dict["optimizer"] == "adam":
        assert(param_dict["epsilon"] == optimizer.epsilon)
        assert(param_dict["amsgrad"] == optimizer.amsgrad)
    # Type check.
    if "batch_size" in param_dict:
        assert(isinstance(param_dict["batch_size"], int))
    if "epochs" in param_dict:
        assert(isinstance(param_dict["epochs"], int))
    if "max_words" in param_dict:
        assert(isinstance(param_dict["max_words"], int))
    if "nunits" in param_dict:
        assert(isinstance(param_dict["nunits"], int))
    if "skip_top" in param_dict:
        assert(isinstance(param_dict["skip_top"], int))

    return
