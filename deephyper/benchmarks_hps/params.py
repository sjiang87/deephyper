# Common hyperparameter definitions.
from hyper2018.api.base import param, interval, step

loss_weights = param.non_ordinal("loss_weights", [list(), dict()])
weighted_metrics = param.non_ordinal("weighted_metrics", [list()])
#batch_size = param.discrete("batch_size", interval(1, 10000), step.ARITHMETIC, 1)
#epochs = param.discrete("epochs", interval(1, TRAINING_SIZE), step.ARITHMETIC, 1)
shuffle = param.non_ordinal("shuffle", [True, False, "batch"])
class_weight = param.non_ordinal("class_weight", [dict()])
# sample_weight = param.non_ordinal("sample_weight", [np.array([])])
#test_size = param.discrete("test_size", interval(INPUT_SIZE, TRAINING_SIZE),
    #step.ARITHMETIC, 1)
hidden_size = param.discrete("hidden_size", interval(0, 30), step.ARITHMETIC, 1)
dropout = param.continuous("dropout", interval(0, 1))
#filters = param.discrete("filter", interval(1, 100), step.ARITHMETIC, 1)
kernel_size = param.discrete("kernal_size", interval(1, 7), step.ARITHMETIC, 1)
padding = param.non_ordinal("padding", ["valid", "same"])
stride = param.discrete("stride", interval(1, 6), step.ARITHMETIC, 1)
#dilation_rate = param.non_ordinal("dilation_rate", ["dilation_rate > 1 & stride == 1",
   # "dilation_rate == 1 & stride > 1"])
data_augmentation = param.non_ordinal('data_augmentation', [False, True])
clipnorm = param.continuous('clipnorm', interval(1e-04, 1))
clipvalue = param.continuous('clipvalue', interval(1e-04, 1))
earlystop = param.non_ordinal('earlystop', ['False', 'True'])
# DO NOT _OPTIMIZE OVER LOSS_FUNCTION
# loss_function = param.non_ordinal("loss_function", ["mean_squared_error", "mean_absolute_error",
#     "mean_squared_percentage_error", "mean_squared_logarithmic_error",
#     "squared_hinge", "hinge", "categorical_hinge", "logcosh",
#     "categorical_crossentropy", "sparse_categorical_crossentropy",
#     "binary_crossentropy", "kullback_leibler_divergence", "poisson",
#     "cosine_proximity"])

optimizer = param.conditional("optimizer", {
    "sgd": [param.continuous("learning_rate", interval(0, 1)),
            param.continuous("decay", interval(0, 1)),
            param.continuous('momentum', interval(0, 1)),
            param.non_ordinal('nesterov', [True, False])],

    "adam": [param.continuous("learning_rate", interval(0, 1)),
            param.continuous("beta_1", interval(0, 1 - 1e-06)),
            param.continuous("beta_2", interval(0, 1 - 1e-08)),
            param.continuous("decay", interval(0, 1)),
            param.continuous("epsilon", interval(1e-20, 1)),
            param.non_ordinal('amsgrad', [True, False])],

    "adamax": [param.continuous("learning_rate", interval(0, 1)),
                param.continuous("beta_1", interval(0, 1 - 1e-06)),
                param.continuous("beta_2", interval(0, 1 - 1e-08)),
                param.continuous("decay", interval(0, 1)),
                param.continuous("epsilon", interval(1e-20, 1))],

    "nadam": [param.continuous("learning_rate", interval(0, 1)),
                param.continuous("beta_1", interval(0, 1 - 1e-06)),
                param.continuous("beta_2", interval(0, 1 - 1e-08)),
                param.continuous("epsilon", interval(1e-20, 1))],

    "rmsprop": [param.continuous('learning_rate', interval(0, 1)),
                param.continuous("rho", interval(0, 1)),
                param.continuous("decay", interval(0, 1)),
                param.continuous("epsilon", interval(1e-20, 1))],

    "adadelta": [param.continuous("rho", interval(0, 10)),
                param.continuous("decay", interval(0, 1)),
                param.continuous("epsilon", interval(1e-20, 1)),
                param.continuous("learning_rate", interval(0, 1))],

    "adagrad": [param.continuous("decay", interval(0, 1)),
                param.continuous("epsilon", interval(1e-20, 1)),
                param.continuous("learning_rate", interval(0, 1))],
})

# DO NOT _OPTIMIZE OVER METRICS
# metrics = param.conditional("metrics", {
#     "binary_accuracy": [],
#     "categorical_accuracy": [],
#     #"sparse_categorical_accuracy": [],
#     "top_k_categorical_accuracy":[param.discrete("k", interval(4, 6), step.ARITHMETIC, 1)],
#     #"sparse_top_k_categorical_accuracy":[],
# })

# callback = param.conditional("callback", {
#     "EarlyStopping": [param.non_ordinal("monitor", [str()]),
#                     param.continuous("baseline", None),
#                     param.discrete("D", interval(1, EPOCH),step.ARITHMETIC, 1),
#                     param.discrete("C", interval(0, EPOCH),step.ARITHMETIC, 1)],
#
#     "LearningRateScheduler": [param.discrete("schedule", interval(0, EPOCH_NUM),                            step.ARITHMETIC, 1)],
#
#     "ReduceLROnPlateau": [param.discrete("cooldown", interval(1, EPOCH_NUM),                                 step.ARITHMETIC, 1),
#                         param.continuous("min_lr", interval(0, 50)),
#                         param.discrete("D", interval(1, EPOCH_NUM),step.ARITHMETIC, 1),
#                         param.discrete("C", interval(0, EPOCH_NUM), step.ARITHMETIC, 1)]
# })

activation = param.conditional("activation", {
    #"softmax": [param.discrete("axis", interval(-5, 5), step.ARITHMETIC, 1)],
    "elu": [param.continuous("alpha", interval(0, 1))],
    "selu": [],
    "softplus": [],
    "softsign": [],
    "relu": [param.continuous("alpha", interval(0, 1))], # max_value can also be specified for relu but we choose not to place a limit.
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

initializer = param.conditional("initializer", {
    "RandomNormal": [param.continuous("mean", interval(0, 100)),
                    param.continuous("stddev", interval(0, 100))],

    "TruncatedNormal": [param.continuous("mean", interval(0, 100)),
                        param.continuous("stddev", interval(0, 100))],

    "RandomUniform": [param.continuous("minval", interval(0, 100)),
                        param.continuous("maxval", interval(0, 100))],

    "VarianceScaling": [param.continuous("scale", interval(0, 100)),
                        param.non_ordinal("mode", ["fan_in", "fan_out", "fan_avg"]),
                        param.non_ordinal("distribution", ["normal", "uniform"])
                        ],

    "Orthogonal": [param.discrete("gain", interval(0, 100), step.ARITHMETIC, 1)],
    "identity": [param.discrete("gain", interval(0, 100), step.ARITHMETIC, 1)],
    "zeros": [],
    "ones": [],
    "lecun_uniform": [],
    "glorot_normal": [],
    "glorot_uniform": [],
    "he_normal": [],
    "lecun_normal": [],
    "he_uniform": []
})
