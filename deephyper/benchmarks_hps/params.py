"""
A module for common hyperparameter definitions.
"""
from deephyper.models.base import param, step

clipnorm = param.continuous('clipnorm', 1e-04, 1)
clipvalue = param.continuous('clipvalue', 1e-04, 1)
data_augmentation = param.non_ordinal('data_augmentation', [False, True])
dropout = param.continuous("dropout", 0, 1, start=0.5)
filter = param.discrete("filter", 1, 100, step.ARITHMETIC, 1)
hidden_size = param.discrete("hidden_size", 0, 30, step.ARITHMETIC, 1)
kernel_size = param.discrete("kernel_size", 1, 7, step.ARITHMETIC, 1)
padding = param.non_ordinal("padding", ["valid", "same"])
shuffle = param.non_ordinal("shuffle", [True, False, "batch"])
stride = param.discrete("stride", 1, 6, step.ARITHMETIC, 1)

# Optimizer hyperparameters
amsgrad = param.non_ordinal("amsgrad", [True, False])
beta_1 = param.continuous("beta_1", 0, 1 - 1e-06)
beta_2 = param.continuous("beta_2", 0, 1 - 1e-08)
decay = param.continuous("decay", 0, 1)
epsilon = param.continuous("epsilon", 1e-20, 1)
learning_rate = param.continuous("learning_rate", 0, 1)
learning_rate_adadelta = param.continuous("learning_rate", 0, 10)
momentum = param.continuous("momentum", 0, 1)
nesterov = param.non_ordinal("nesterov", [True, False])
rho = param.continuous("rho", 0, 1)

optimizer = param.conditional("optimizer", {
    "sgd": [learning_rate, momentum, decay, nesterov],
    "rmsprop": [learning_rate, rho, epsilon, decay],
    "adagrad": [learning_rate, epsilon, decay],
    "adadelta": [learning_rate_adadelta, rho, epsilon, decay],
    "adam": [learning_rate, beta_1, beta_2, epsilon, decay, amsgrad],
    "adamax": [learning_rate, beta_1, beta_2, epsilon, decay],
    # schedule_decay can also be specified for nadam but we choose
    # not to optimize over that parameter.
    "nadam": [learning_rate, beta_1, beta_2, epsilon]
})

# Activation function hyperparameters
alpha = param.continuous("alpha", 0, 1, start=0)
alpha1 = param.continuous("alpha1", 0, 1, start=0)
alpha2 = param.continuous("alpha2", 0, 1, start=0)
alpha3 = param.continuous("alpha3", 0, 1, start=0)
alpha4 = param.continuous("alpha4", 0, 1, start=0)
alpha5 = param.continuous("alpha5", 0, 1, start=0)

activation = param.conditional("activation", {
    "elu": [alpha],
    "selu": [],
    "softplus": [],
    "softsign": [],
    # max_value can also be specified for relu but we choose
    # not to optimize over that parameter.
    "relu": [alpha],
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})
activation1 = param.conditional("activation1", {
    "elu": [alpha],
    "selu": [],
    "softplus": [],
    "softsign": [],
    # max_value can also be specified for relu but we choose
    # not to optimize over that parameter.
    "relu": [alpha],
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})
activation2 = param.conditional("activation2", {
    "elu": [alpha],
    "selu": [],
    "softplus": [],
    "softsign": [],
    # max_value can also be specified for relu but we choose
    # not to optimize over that parameter.
    "relu": [alpha],
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})
activation3 = param.conditional("activation3", {
    "elu": [alpha],
    "selu": [],
    "softplus": [],
    "softsign": [],
    # max_value can also be specified for relu but we choose
    # not to optimize over that parameter.
    "relu": [alpha],
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})
activation4 = param.conditional("activation4", {
    "elu": [alpha],
    "selu": [],
    "softplus": [],
    "softsign": [],
    # max_value can also be specified for relu but we choose
    # not to optimize over that parameter.
    "relu": [alpha],
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})
activation5 = param.conditional("activation5", {
    "elu": [alpha],
    "selu": [],
    "softplus": [],
    "softsign": [],
    # max_value can also be specified for relu but we choose
    # not to optimize over that parameter.
    "relu": [alpha],
    "tanh": [],
    "sigmoid": [],
    "hard_sigmoid": [],
    "linear": []
})

# Flat versions of conditional parameters.
optimizer_flat = param.non_ordinal("optimizer",
    ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"])
optimizer_flat_params = [optimizer_flat, amsgrad, beta_1, beta_2, decay,
    epsilon, learning_rate, momentum, nesterov, rho]

activation_flat = param.non_ordinal("activation",
    ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
     "hard_sigmoid", "linear"])
activation_flat_params = [activation_flat, alpha]

activation1_flat = param.non_ordinal("activation1",
    ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
     "hard_sigmoid", "linear"], start='relu')
activation1_flat_params = [activation1_flat, alpha1]

activation2_flat = param.non_ordinal("activation2",
    ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
     "hard_sigmoid", "linear"], start='relu')
activation2_flat_params = [activation2_flat, alpha2]

activation3_flat = param.non_ordinal("activation3",
    ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
     "hard_sigmoid", "linear"], start='relu')
activation3_flat_params = [activation3_flat, alpha3]

activation4_flat = param.non_ordinal("activation4",
    ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
     "hard_sigmoid", "linear"], start='relu')
activation4_flat_params = [activation4_flat, alpha4]

activation5_flat = param.non_ordinal("activation5",
    ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
     "hard_sigmoid", "linear"], start='relu')
activation5_flat_params = [activation5_flat, alpha5]
