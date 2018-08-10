from deephyper.search.models import Type

class ParameterType(Type):
    """A class to specify the types of hyperparameters."""
    CONTINUOUS  = 1
    DISCRETE    = 2
    NON_ORDINAL = 3
    CONDITIONAL = 4
