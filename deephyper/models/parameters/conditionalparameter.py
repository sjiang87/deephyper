from collections import OrderedDict

from deephyper.models.parameter import Parameter
from deephyper.models.types.parametertype import ParameterType

class ConditionalParameter(Parameter):
    """A class to describe a hyperparameter that exposes other
    hyperparameters depending on its value. Its values will be treated
    as non-ordinal.
    """

    def __init__(self, name, branches, start=None):
        """
        Keyword arguments:
        name (str) -- A string to identify the parameter.
        branches (dict) -- A dictionary where keys are values taken on by this
                           parameter, and values are lists of parameter objects
                           exposed when that value is taken on.
        start (any) -- The starting point for evaluation on this
                         hyperparameter. Defaults to the first key in `branches`
                         if none is specified.
        """
        self.branches = branches
        if start is None:
            start = list(branches.keys())[0]
        super(ConditionalParameter, self).__init__(name,
                                                   ParameterType.CONDITIONAL,
                                                   start)
        # Make branches an OrderedDict for convenience.
        self.branches = OrderedDict(branches)

        return

    # Provide a convenient way to output information about the parameter.
    def __str__(self):
        return ("<param n: {0}, t: {1}, branch keys: {2}>".format(
                self.name, self.type, self.branches.keys()))

    def __repr__(self):
        return self.__str__()

    def debug(self):
        """Ensure that the parameter was constructed properly."""
        super(ConditionalParameter, self).debug()

        # Check that branches is a dictionary.
        if not isinstance(self.branches, dict):
            raise Exception("Branches attribute of a conditional parameter "
                            "must be a dictionary: {0}".format(self))

        # Check validity of branches object.
        # Note that it is not possible to have duplicate keys in a python
        # dictionary because python will only accept one of the keys attempted
        # to be inserted. Therefore, conditional parameters should be constructed
        # with distinct branch keys in the branches dictionary, per depth level.
        for branch_name, param_list in self.branches.items():
            # Ensure that each branch exposes a list.
            if not isinstance(param_list, list):
                raise Exception("Branches attribute of a conditional parameter "
                    "must have list type values at branch key {0} of "
                    "parameter: {1}.".format(branch_name, self))
            else:
                # Ensure that the value of each branch key is a list of parameters.
                for index, param in enumerate(param_list):
                    if not isinstance(param, Parameter):
                        raise Exception("Branches attribute of a conditional"
                            "parameter must specify lists of parameter "
                            "objects. Not the case for index: {0}, "
                            "branch key: {1}, parameter: {2}".format(
                            index, branch_name, self))

        return
