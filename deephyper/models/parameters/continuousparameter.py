from deephyper.models.parameter import Parameter
from deephyper.models.types.parametertype import ParameterType
from deephyper.models.types.priortype import PriorType

class ContinuousParameter(Parameter):
    """
    A class to describe a hyperparameter that takes continuous values
    on a numerical interval.
    """

    def __init__(self, name, low, high, prior=PriorType.UNIFORM, start=None):
        """
        Keyword arguments:
        name (str) -- A string to identify the parameter.
        low (numeric) -- The lower bound of the parameter's value interval
                         (inclusive).
        high (numeric) -- The upper bound of the parameter's value interval
                          (inclusive).
        prior (PriorType) -- Determines how samples from the interval are
                             distributed.
        start (any) -- The starting point for evaluation on this hyperparameter.
                         Defaults to 'low' if not specified.
        """
        # Implementation note: Optimizers may implement the same priors
        # differently. We choose to not internally ensure that distributions
        # span the same effective interval. For instance, Hyperopt implements
        # loguniform with the natural logarithm, SKOpt with the
        # base 10 logarithm.
        self.low = low
        self.high = high
        self.prior = prior
        if start is None:
            start = low
        super(ContinuousParameter, self).__init__(name,
                                                  ParameterType.CONTINUOUS,
                                                  start)

        return

    # Provide a convenient way to output information about the parameter.
    def __str__(self):
        return ("<param n: \'{0}\', t: {1}, low: {2}, high: {3}, prior: {4}>"
                .format(self.name, self.type, self.low, self.high, self.prior))

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
