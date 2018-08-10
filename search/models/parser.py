class Parser:
    """
    A class to handle transforming fundamental hyperparameters and spaces
    to hyperparameters and spaces in the format of a given optimizer.
    """

    # Assume that a parser does not support conditional branching unless
    # otherwise specified.
    supports_conditionals = False

    @classmethod
    def transform_param(cls, param):
        """
        Transform a fundamental parameter to a parameter in the format
        of a given optimizer.
        """
        pass

    @classmethod
    def transform_space(cls, space):
        """
        From a list of fundamental parameters, generate a space that
        conforms to an optimizer's format.
        """
        pass
