"""
This module allows for a nice parameter construction syntax.
e.g. 'param.continuous("foo", 1, 5)'
"""
from __future__ import absolute_import

# Import each hyperparameter class.
from deephyper.search.models.parameters.conditionalparameter import \
     ConditionalParameter as conditional
from deephyper.search.models.parameters.continuousparameter import \
     ContinuousParameter as continuous
from deephyper.search.models.parameters.discreteparameter import \
     DiscreteParameter as discrete
from deephyper.search.models.parameters.nonordinalparameter import \
     NonOrdinalParameter as non_ordinal
