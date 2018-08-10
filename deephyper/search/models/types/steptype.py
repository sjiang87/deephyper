from deephyper.search.models.type import Type

class StepType(Type):
    """
    A class to specify the ways in which an interval can be traversed
    for discrete values.
    """
    ARITHMETIC = 1
    GEOMETRIC  = 2
