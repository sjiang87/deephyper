class _Problem(object):
    """
    A class to represent a hyperparameter optimization problem.

    Attributes:
    space (list) -- A list of `Parameter` objects that make up the decision space.
    params (list) -- A list of the names of the hyperparameters in the space.
    starting point (list) -- A list of values that should be used for the first
                             evaluation of the problem.
    """
    def __init__(self, space, starting_point=None):
        """
        space (list) -- See class attributes.
        starting_point (list) -- See class attributes. If one is not specified,
                                 default values will be used.
        """
        super(_Problem, self).__init__()
        self.space = space
        self.params = [param.name for param in space]
        if starting_point is None:
            starting_point = [param.start for param in space]
        self.starting_point = starting_point

        return

    def __str__(self):
        """Display parameter names and starting point."""

        return "{0}\n{1}".format(super(_Problem, self).__str__(),
                                 list(zip(self.params, self.starting_point)))
