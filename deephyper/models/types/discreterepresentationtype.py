from deephyper.models.type import Type

class DiscreteRepresentationType(Type):
    """
    A class to specify how a discrete interval should be represented to
    the optimizer.
    """
    ORDINAL = 1
    DEFAULT = 2
