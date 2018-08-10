from skopt.space.space import Real, Integer, Categorical, Space
from deephyper.search.models.parser import Parser
from deephyper.search.models.types.steptype import StepType
from deephyper.search.models.types.priortype import PriorType
from deephyper.search.models.types.parametertype import ParameterType
from deephyper.search.models.types.discreterepresentationtype import DiscreteRepresentationType

class SKOptParser(Parser):
    """
    A class to transform hyperparameters and spaces into SKOpt's format
    for hyperparameters and spaces.
    """

    # SKOpt does not natively support parameters of the conditional type.
    conditional_support = False

    @classmethod
    def transform_param(cls, param):
        """
        Generate a parameter that conforms to SKOpt's format.

        Keyword arguments:
        param -- An object of type `Parameter` that will be transformed
                 into an SKOpt dimension.
        """
        param_name = param.name
        param_type = param.type

        if param_type == ParameterType.CONTINUOUS:
            if param.prior == PriorType.UNIFORM:
                prior = "uniform"
            elif param.prior == PriorType.LOGUNIFORM:
                prior = "loguniform"
            return Real(low=param.low, high=param.high, prior=prior,
                        name=param_name)

        # SKOpt only has native support for discrete parameters with an
        # arithmetic step type and step size of 1.
        elif param_type == ParameterType.DISCRETE:
            drt = param.drt
            step_size = param.step_size
            step_type = param.step_type

            if step_type == StepType.ARITHMETIC and step_size == 1:
                return Integer(low=param.low, high=param.high, name=param_name)

            elif drt == DiscreteRepresentationType.ORDINAL:
                return Integer(low=0, high=param.max_n, name=param_name)

            elif drt == DiscreteRepresentationType.DEFAULT:
                return Categorical(categories=param.interval_list,
                                   name=param_name)

        elif param_type == ParameterType.NON_ORDINAL:
            return Categorical(categories=param.values, name=param_name)

        elif param_type == ParameterType.CONDITONAL:
            raise Exception("A conditional parameter was found while parsing "
                            "for SKOpt. SKOpt does not support conditional "
                            "parameters. {0}".format(param))

    @classmethod
    def transform_space(cls, params):
        """
        Generate a space that conforms to SKOpt's format.

        Keyword arguments:
        params -- An iterable that contains parameter objects.
        """
        dimensions = []
        for param in params:
            dimensions.append(cls.transform_param(param))

        skopt_space = Space(dimensions)

        return skopt_space
