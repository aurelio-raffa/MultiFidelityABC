import numpy as np

from source.models.base_model import ForwardModel


class HighFidelityModel(ForwardModel):
    """Base class implementing a wrapper to a high fidelity model.

    This class represents the high fidelity model that in our case coincides
        with the true forward model. We build methods to evaluate the forward
        model at a given parameter and the (log-) posterior of the parameter.
    """
    def __init__(self, core_function, data, evaluation_nodes, log_error_density, log_prior):
        """Parameters
        ----------
        core_function : function or callable object
            A function returning the evaluations of the forward model for a given value of the parameter(s)
            evaluated in the (fixed) set of spatial nodes (potentially passed as parameters).
        data : numpy.ndarray
            The sample of observed values of the forward model in the spatial nodes.
        evaluation_nodes : numpy.ndarray
            Matrix of the spatial nodes at which the data has been observed (every column is a node).
        log_error_density : function or callable object
            A function that returns the joint log density of the errors on a vector of values.
        log_prior : function or callable object
            It computes the log prior at the input vector (of parameters).
        """
        super().__init__(data, log_error_density, log_prior)
        self.core_function = core_function
        self.evaluation_nodes = evaluation_nodes

    def _eval_subroutine(self, z):
        return self.core_function(z, self.evaluation_nodes)
