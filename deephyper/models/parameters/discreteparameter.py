from math import log, floor

from deephyper.models.parameter import Parameter
from deephyper.models.types.discreterepresentationtype import \
    DiscreteRepresentationType
from deephyper.models.types.parametertype import ParameterType
from deephyper.models.types.steptype import StepType



class DiscreteParameter(Parameter):
    """
    A class to describe a hyperparameter that takes discrete values
    on a numerical interval.
    """

    def __init__(self, name, low, high, step_type=StepType.ARITHMETIC,
                 step_size=1, drt=DiscreteRepresentationType.DEFAULT,
                 map_negative=False, start=None):
        """
        Keyword arguments:
        name (str) -- A string to identify the parameter.
        low (numeric) -- The lower bound of the interval on which the parameter
                         takes values (inclusive).
        high (numeric) -- The upper bound of the interval on which the parameter
                          takes values (inclusive).
        step_type (StepType) -- Specifies the way in which the parameter's
                                interval is traversed, e.g. arithmetic.
        step_size (numeric) -- The magnitude of each step taken on the
                               parameter's interval.
        drt (DiscreteRepresentationType)-- Specifies how the parameter should
                                           be presented to a hyperparameter
                                           optimizer.
        map_negative (bool) -- If the discrete interval has a geometric step
                               type and its values should be negative, set this
                               flag and specify positive values for low, high,
                               and step_size.
        start (any) -- The starting point for evaluation on this hyperparameter.
                         Defaults to 'low' if not specified.
        """
        # Implementation note: For geometric parameters with negative intervals,
        # it is easier to invert the interval.
        # E.g. [-8, -4, -2, -1] has step size 0.5, but we prefer to specify
        # the interval as [1, 2, 4, 8] with step size 2 and adding negative
        # signs where necessary via `map_negative`.
        self.low = low
        self.high = high
        self.step_type = step_type
        self.step_size = step_size
        self.drt = drt
        self.map_negative = map_negative
        if start is None:
            start = low
        super(DiscreteParameter, self).__init__(name, ParameterType.DISCRETE,
                                                start)

    # Provide a convenient way to output information about the parameter.
    def __str__(self):
        return ("<param n: \'{0}\', t: {1}, low: {2}, high: {3}, step_t: {4}, "
                "step_s: {5}, drt: {6}, is_neg: {7}>".format(
                self.name, self.type, self.low, self.high, self.step_type,
                self.step_size, self.drt, self.map_negative))

    def __repr__(self):
        return self.__str__()

    @property
    def interval_list(self):
        """
        Return a list of values that are on the discrete interval
        of the parameter.
        """
        # Unpack values for frequent loop reference.
        high = self.high
        map_neg = self.map_negative
        step_type = self.step_type
        step_size = self.step_size
        values = list()
        value_cur = self.low

        # Step through each value on the discrete interval and add it to
        # the values list.
        while value_cur <= high:
            # Add current value to values.
            if map_neg:
                values.append(-value_cur)
            else:
                values.append(value_cur)
            # Step to next value on the interval.
            if step_type == StepType.ARITHMETIC:
                value_cur += step_size
            elif step_type == StepType.GEOMETRIC:
                value_cur *= step_size

        return values

    # Implementation note: The approach of defining the function
    # 'map_to_interval` to map {0, 1, 2, 3, ..., n_max} to the parameter's
    # interval could be replaced by storing a list of the values on the
    # parameter's interval and using the non-negative sequence
    # {0, 1, ..., n_max} to index the list. This approach would require more
    # memory, and would still yield a constant time algorithim.
    # Thus the 'map_to_interval' approach was chosen.
    def map_to_interval(self, n):
        """Return the nth value on the parameter's interval (0-indexed)."""
        if self.step_type == StepType.ARITHMETIC:
            abs_val = self.low + (self.step_size * n)
        elif self.step_type == StepType.GEOMETRIC:
            abs_val = self.low * (self.step_size ** n)

        if self.map_negative:
            return -abs_val
        else:
            return abs_val

    @property
    def max_n(self):
        """
        Return the greatest n, element of the naturals, such that:
            Arithmetic: low + (step_size * n) <= high.
            Geometric:  low * (step_size ^ n) <= high.
        i.e. what is the largest value that should be passed to map_to_interval?
        """
        # Unpack and cast parameter values for correct arithmetic.
        low = float(self.low)
        high = float(self.high)
        step_size = float(self.step_size)

        # Compute max n.
        if self.step_type == StepType.ARITHMETIC:
            n = int(floor((high - low) / step_size))
            # Correct for roundoff error.
            if (n * step_size + low) <= (high - step_size):
                return n + 1
            else:
                return n
        elif self.step_type == StepType.GEOMETRIC:
            if low < 0:
                n = int(floor(log(low / high, step_size)))
                # Correct for roundoff error.
                if(low / (step_size ** n) <= high * step_size):
                    return n + 1
                else:
                    return n
            else:
                n = int(floor(log(high / low, step_size)))
                # Correct for roundoff error.
                if (low * (step_size ** n)) <= (high / step_size):
                    return n + 1
                else:
                    return n

    # Ensure the parameter was constructed properly.
    def debug(self):
        """Ensure that the parameter's construction was well-formed."""
        super(DiscreteParameter, self).debug()

        # Ensure valid step type.
        if not isinstance(self.step_type, StepType):
            raise Exception("Parameter constructed with unrecognized step "
                            "type: {0}".format(self))

        # Ensure valid representation type.
        if not isinstance(self.drt, DiscreteRepresentationType):
            raise Exception("Parameter constructed with unrecognized "
                            "representation type: {0}".format(self))

        # Ensure the interval has valid lower and upper bounds.
        if (self.low < 0
                and self.step_type == StepType.GEOMETRIC
                and not self.map_negative):
            raise Exception("Discrete parameter constructed with negative lower "
                            "bound. Please make use of the `map_negative` "
                            "constructor argument for geometric intervals. "
                            "{0}".format(self))

        if self.low >= self.high:
            raise Warning("Parameter's lower bound exceeds or is equal to "
                          "to its upper bound: {0}".format(self))

        # Ensure a valid step size was given.
        if self.step_size <= 0:
            raise Warning("Parameter constructed with step size less than "
                          "or equal to 0: {0}".format(self))

        if (self.step_type == StepType.GEOMETRIC
                and self.step_size <= 1):
            raise Exception("Parameter has geometric step type and step size "
                            "less than or equal to 1: {0}".format(self))

        # Check for miscellaneous inconsistencies.
        # Ensure the upper and lower bounds of a discrete, geometric parameter
        # are not zero and that the signs of the bounds match.
        if self.step_type == StepType.GEOMETRIC:
            if self.low == 0 or self.high == 0:
                raise Warning("Parameter has geometric step type and a bound "
                              "of 0 on its interval: {0}".format(self))
            if ((self.low < 0 and self.high > 0)
                    or (self.low > 0 and self.high < 0)):
                raise Warning("Parameter has geometric step type and its "
                              "bounds have different sign: {0}".format(self))

        return
