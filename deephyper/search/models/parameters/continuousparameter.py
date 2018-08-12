from deephyper.search.models.parameter import Parameter
from deephyper.search.models.types.priortype import PriorType
from deephyper.search.models.types.parametertype import ParameterType

class ContinuousParameter(Parameter):
    """
    A class to describe a hyperparameter that takes continuous values
    on a numerical interval.
    """

    def __init__(self, name, low, high, prior=PriorType.UNIFORM):
        """
        Keyword arguments:
        name -- A string to identify the parameter.
        low -- The lower bound of the parameter's value interval (inclusive).
        high -- The upper bound of the parameter's value interval (inclusive).
        prior -- A `PriorType` that specifies how distributions are sampled
                 from the interval.
        """
        # Implementation note: Optimizers may implement the same priors
        # differently. We choose to not internally ensure that distributions
        # span the same effective interval. For instance, Hyperopt implements
        # loguniform with the natural logarithm, SKOpt with the
        # base 10 logarithm.
        self.low = low
        self.high = high
        self.prior = prior
        super(ContinuousParameter, self).__init__(name,
                                                  ParameterType.CONTINUOUS)

        return

    # Provide a convenient way to output information about the parameter.
    def __str__(self):
        return ("<param n: \'{0}\', t: {1}, low: {2}, high: {3}, prior: {4}>".format(
                self.name, self.type, self.low, self.high, self.prior))

    def __repr__(self):
        return self.__str__()

    def debug(self):
        """Ensure that the parameter was constructed properly."""
        super(ContinuousParameter, self).debug()

        # Ensure the interval has valid lower and upper bounds.
        if self.low >= self.high:
            raise Warning("Parameter's lower bound exceeds or is equal to "
                          "its upper bound: {0}".format(self))

        return
