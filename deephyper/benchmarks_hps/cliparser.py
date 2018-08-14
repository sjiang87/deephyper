from argparse import ArgumentParser
from deephyper.benchmarks_hps.util import str2bool

def build_base_parser():
    """A command line parser for keras."""
    # Note that each benchmark implements its own cli parser on top of this one.

    parser = ArgumentParser(add_help=True)

    # Hyperparameters.
    parser.add_argument("--activation", action="store",
                        dest="activation",
                        nargs="?", const=1, type=str, default="relu",
                        choices=["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
                                 "hard_sigmoid", "linear", "LeakyReLU", "PReLU", "ELU", "ThresholdedReLU"],
                        help="type of activation function hidden layer")

    parser.add_argument("--batch_size", action="store", dest="batch_size", type=int, default=128, help="batch size")

    parser.add_argument("--clipnorm", action="store", dest="clipnorm",
                        nargs="?", const=1, type=float, default=1.0,
                        help="float >= 0. Gradients will be clipped when their \
                        L2 norm exceeds this value.")

    parser.add_argument("--clipvalue", action="store", dest="clipvalue",
                        nargs="?", const=1, type=float, default=0.5,
                        help="float >= 0. Gradients will be clipped when their \
                        absolute value exceeds this value.")

    parser.add_argument("--delta", action="store", dest="delta",
                        nargs="?", const=1, type=float, default=0.0001,
                        help="float >= 0. min delta for early stopping")

    parser.add_argument("--dropout", action="store", dest="dropout", nargs="?", const=1, type=float, default=0.0,
                        help=" float [0, 1). Fraction of the input units to drop")

    parser.add_argument("--epochs", action="store", dest="epochs",
                        nargs="?", const=2, type=int, default=2,
                        help="number of epochs")

    parser.add_argument("--init", action="store", dest="init",
                        nargs="?", const=1, type=str, default="normal",
                        choices=["Zeros", "Ones", "Constant", "RandomNormal", "RandomUniform", "TruncatedNormal",
                                 "VarianceScaling", "Orthogonal", "Identity", "lecun_uniform", "glorot_normal",
                                 "glorot_uniform", "he_normal", "lecun_normal", "he_uniform"],
                        help="type of initialization")

    parser.add_argument("--optimizer", action="store",
                        dest="optimizer",
                        nargs="?", const=1, type=str, default="sgd",
                        choices=["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"],
                        help="type of optimizer")

    # Optimizer hyperparameters.
    parser.add_argument("--learning_rate", action="store", dest="learning_rate",
                        nargs="?", const=1, type=float, default=0.01,
                        help="float >= 0. Learning rate")

    parser.add_argument("--momentum", action="store", dest="momentum",
                        nargs="?", const=1, type=float, default=0.0,
                        help="float >= 0. Parameter updates momentum")

    parser.add_argument("--decay", action="store", dest="decay",
                        nargs="?", const=1, type=float, default=0.0,
                        help="float >= 0. Learning rate decay over each update")

    parser.add_argument("--nesterov", action="store", dest="nesterov",
                        nargs="?", const=1, type=str2bool, default=False,
                        help="boolean. Whether to apply Nesterov momentum?")

    parser.add_argument("--rho", action="store", dest="rho",
                        nargs="?", const=1, type=float, default=0.9,
                        help="float >= 0")

    parser.add_argument("--epsilon", action="store",
                        dest="epsilon",
                        nargs="?", const=1, type=float, default=1e-08,
                        help="float >= 0")

    parser.add_argument("--beta_1", action="store", dest="beta_1",
                        nargs="?", const=1, type=float, default=0.9,
                        help="float >= 0")

    parser.add_argument("--beta_2", action="store", dest="beta_2",
                        nargs="?", const=1, type=float, default=0.999,
                        help="float >= 0")

    parser.add_argument("--amsgrad", action="store", dest="amsgrad",
                        nargs="?", const=1, type=str2bool, default=False,
                        help=" boolean. Whether to apply the AMSGrad variant of this algorithm [Adam] from the paper \"On the Convergence of Adam and Beyond\".")

    # Activation hyperparameters.
    parser.add_argument("--alpha", action="store", dest="alpha",
                        type=float, default=1.0)

    # Misc options.
    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s 0.1")

    parser.add_argument("--backend", action="store",
                        dest="backend",
                        nargs="?", const=1, type=str, default="tensorflow",
                        choices=["tensorflow", "theano", "cntk"],
                        help="Keras backend")

    parser.add_argument("--model_path", help="path from which models are loaded/saved", type=str, default="")

    parser.add_argument("--data_source", help="location of dataset to load", type=str, default="")

    parser.add_argument("--stage_in_destination", help="if provided; cache data at this location",
                        type=str, default="")

    parser.add_argument("--timeout", help="benchmark timeout in mins. defaults to 0 which indicates no timeout.",
                        action="store", dest="timeout",
                        nargs="?", const=1, type=float,
                        default=0.0)

    return parser
