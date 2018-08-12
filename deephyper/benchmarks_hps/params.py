# Common hyperparameter definitions.
from deephyper.search.models.base import param, step

loss_weights = param.non_ordinal("loss_weights", [list(), dict()])
weighted_metrics = param.non_ordinal("weighted_metrics", [list()])
#batch_size = param.discrete("batch_size", 1, 10000), step.ARITHMETIC, 1)
#epochs = param.discrete("epochs", 1, TRAINING_SIZE), step.ARITHMETIC, 1)
shuffle = param.non_ordinal("shuffle", [True, False, "batch"])
class_weight = param.non_ordinal("class_weight", [dict()])
# sample_weight = param.non_ordinal("sample_weight", [np.array([])])
#test_size = param.discrete("test_size", INPUT_SIZE, TRAINING_SIZE),
    #step.ARITHMETIC, 1)
hidden_size = param.discrete("hidden_size", 0, 30, step.ARITHMETIC, 1)
dropout = param.continuous("dropout", 0, 1)
#filters = param.discrete("filter", 1, 100), step.ARITHMETIC, 1)
kernel_size = param.discrete("kernal_size", 1, 7, step.ARITHMETIC, 1)
padding = param.non_ordinal("padding", ["valid", "same"])
stride = param.discrete("stride", 1, 6, step.ARITHMETIC, 1)
#dilation_rate = param.non_ordinal("dilation_rate", ["dilation_rate > 1 & stride == 1",
   # "dilation_rate == 1 & stride > 1"])
data_augmentation = param.non_ordinal('data_augmentation', [False, True])
clipnorm = param.continuous('clipnorm', 1e-04, 1)
clipvalue = param.continuous('clipvalue', 1e-04, 1)
earlystop = param.non_ordinal('earlystop', ['False', 'True'])
# DO NOT _OPTIMIZE OVER LOSS_FUNCTION
# loss_function = param.non_ordinal("loss_function", ["mean_squared_error", "mean_absolute_error",
#     "mean_squared_percentage_error", "mean_squared_logarithmic_error",
#     "squared_hinge", "hinge", "categorical_hinge", "logcosh",
#     "categorical_crossentropy", "sparse_categorical_crossentropy",
#     "binary_crossentropy", "kullback_leibler_divergence", "poisson",
#     "cosine_proximity"])

optimizer = param.conditional("optimizer", {
    "sgd": [param.continuous("learning_rate", 0, 1),
            param.continuous("decay", 0, 1),
            param.continuous('momentum', 0, 1),
            param.non_ordinal('nesterov', [True, False])],

    "adam": [param.continuous("learning_rate", 0, 1),
            param.continuous("beta_1", 0, 1 - 1e-06),
            param.continuous("beta_2", 0, 1 - 1e-08),
            param.continuous("decay", 0, 1),
            param.continuous("epsilon", 1e-20, 1),
            param.non_ordinal('amsgrad', [True, False])],

    "adamax": [param.continuous("learning_rate", 0, 1),
                param.continuous("beta_1", 0, 1 - 1e-06),
                param.continuous("beta_2", 0, 1 - 1e-08),
                param.continuous("decay", 0, 1),
                param.continuous("epsilon", 1e-20, 1)],

    "nadam": [param.continuous("learning_rate", 0, 1),
                param.continuous("beta_1", 0, 1 - 1e-06),
                param.continuous("beta_2", 0, 1 - 1e-08),
                param.continuous("epsilon", 1e-20, 1)],

    "rmsprop": [param.continuous('learning_rate', 0, 1),
                param.continuous("rho", 0, 1),
                param.continuous("decay", 0, 1),
                param.continuous("epsilon", 1e-20, 1)],

    "adadelta": [param.continuous("rho", 0, 10),
                param.continuous("decay", 0, 1),
                param.continuous("epsilon", 1e-20, 1),
                param.continuous("learning_rate", 0, 1)],

    "adagrad": [param.continuous("decay", 0, 1),
                param.continuous("epsilon", 1e-20, 1),
                param.continuous("learning_rate", 0, 1)],
})

# DO NOT _OPTIMIZE OVER METRICS
# metrics = param.conditional("metrics", {
#     "binary_accuracy": [],
#     "categorical_accuracy": [],
#     #"sparse_categorical_accuracy": [],
#     "top_k_categorical_accuracy":[param.discrete("k", 4, 6), step.ARITHMETIC, 1)],
#     #"sparse_top_k_categorical_accuracy":[],
# })

# callback = param.conditional("callback", {
#     "EarlyStopping": [param.non_ordinal("monitor", [str()]),
#                     param.continuous("baseline", None),
#                     param.discrete("D", 1, EPOCH),step.ARITHMETIC, 1),
#                     param.discrete("C", 0, EPOCH),step.ARITHMETIC, 1)],
#
#     "LearningRateScheduler": [param.discrete("schedule", 0, EPOCH_NUM),                            step.ARITHMETIC, 1)],
#
#     "ReduceLROnPlateau": [param.discrete("cooldown", 1, EPOCH_NUM),                                 step.ARITHMETIC, 1),
#                         param.continuous("min_lr", 0, 50),
#                         param.discrete("D", 1, EPOCH_NUM),step.ARITHMETIC, 1),
#                         param.discrete("C", 0, EPOCH_NUM), step.ARITHMETIC, 1)]
# })

activation = param.conditional("activation", {
    #"softmax": [param.discrete("axis", -5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha", 0, 1)],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha", 0, 1)], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

activation1 = param.conditional("activation1", {
    #"softmax": [param.discrete("axis", -5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha1", 0, 1)],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha1", 0, 1)], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

activation2 = param.conditional("activation2", {
    #"softmax": [param.discrete("axis", -5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha2", 0, 1)],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha2", 0, 1)], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

activation3 = param.conditional("activation3", {
    #"softmax": [param.discrete("axis", -5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha3", 0, 1)],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha3", 0, 1)], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

activation4 = param.conditional("activation4", {
    #"softmax": [param.discrete("axis", -5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha4", 0, 1)],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha4", 0, 1)], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

activation5 = param.conditional("activation5", {
    #"softmax": [param.discrete("axis", -5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha5", 0, 1)],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha5", 0, 1)], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

initializer = param.conditional("initializer", {
    "RandomNormal": [param.continuous("mean", 0, 100),
                    param.continuous("stddev", 0, 100)],

    "TruncatedNormal": [param.continuous("mean", 0, 100),
                        param.continuous("stddev", 0, 100)],

    "RandomUniform": [param.continuous("minval", 0, 100),
                        param.continuous("maxval", 0, 100)],

    "VarianceScaling": [param.continuous("scale", 0, 100),
                        param.non_ordinal("mode", ["fan_in", "fan_out", "fan_avg"]),
                        param.non_ordinal("distribution", ["normal", "uniform"])
                        ],

    "Orthogonal": [param.discrete("gain", 0, 100, step.ARITHMETIC, 1)],
    "identity": [param.discrete("gain", 0, 100, step.ARITHMETIC, 1)],
    "zeros": [],
    "ones": [],
    "lecun_uniform": [],
    "glorot_normal": [],
    "glorot_uniform": [],
    "he_normal": [],
    "lecun_normal": [],
    "he_uniform": []
})
