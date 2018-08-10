from deephyper.search.models.parameter import Parameter
from deephyper.search.models.types.parametertype import ParameterType

class NonOrdinalParameter(Parameter):
    """
    A class to describe a hyperparameter that takes values on which no
    ordering is defined.
    """

    def __init__(self, name, values):
        """
        Keyword arguments:
        name -- A string to identify the parameter.
        values -- A list of values that the parameter takes.
        """
        self.values = values
        super(NonOrdinalParameter, self).__init__(name,
                                                  ParameterType.NON_ORDINAL)

        return

    # Provide a convenient way to display information about the parameter.
    def __str__(self):
        return ("<param n: \'%s\', t: %s, vals: %s"
                % self.name, self.type, self.values)

    def __repr__(self):
        return self.__str__()

    def debug(self):
        """Ensure that the parameter's construction was well-formed."""
        super(NonOrdinalParameter, self).debug()

        # self.values should be of type 'list' because most hyperparameter
        # optimizers require that.
        if not isinstance(self.values, list):
            raise Warning("Parameter of non-ordinal type has values attribute"
                          "that is not a list type. For maximum compliance"
                          "with hyperparameter optimizers, please specify"
                          "non-ordinal values in a list. %s" % self)

        return
