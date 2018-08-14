class Parser:
    """
    A class to handle transforming fundamental hyperparameters and spaces
    to hyperparameters and spaces in the format of a given optimizer.
    """

    # Assume that a parser does not support conditional branching unless
    # otherwise specified.
    supports_conditionals = False
