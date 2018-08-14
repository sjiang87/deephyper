from deephyper.models.types.parametertype import ParameterType

class Parameter(object):
    """
    A class to represent a hyperparmeter.

    Attributes:
    name (str) -- A string that identifies the parameter.
    parameter_type (ParameterType) -- Specifies the type of the parameter.
    start (any) -- The starting point for evaluation on this hyperparameter.
    """

    def __init__(self, name, parameter_type, start):
        """
        Keyword arguments:
        name (str) -- See class attributes.
        parameter_type (ParameterType) -- See class attributes.
        start (any) -- See class attributes.
        """
        super(Parameter, self).__init__()
        self.name = name
        self.type = parameter_type
        self.start = start
        self.debug()

        return

    # Provide a convenient way to output information about the parameter.
    def __str__(self):
        return "<param n:\'{0}\', t: {1}>".format(self.name, self.type)

    def __repr__(self):
        return self.__str__()

    def debug(self):
        """Ensure that the parameter was constructed properly."""
        # Check that the type of the parameter is valid.
        if not isinstance(self.type, ParameterType):
            raise Warning("Parameter constructed with an unrecognized"
                          "type: {0}".format(self))

        return
