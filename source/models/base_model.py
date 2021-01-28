import numpy as np


class ForwardModel:
    """Class implementing the general forward-model paradigm of use.
    """
    def __init__(self, data, log_error_density, log_prior):
        """Parameters
        ----------
        data : numpy.ndarray
            Sample of observed values of the forward model in the spatial nodes.
        log_error_density : function or callable object
            It computes the log density at the input vector (used for computing the likelihood, passing
            as input the difference between the data and the vector of the evaluations of the surrogate model
            in the corresponding spatial nodes for a fixed  parameter z).
        log_prior : function or callable object
            It computes the log prior at the input vector (of parameters).
        """
        self.data = data
        self.log_error_density = log_error_density
        self.log_prior = log_prior
        self.last_evaluations = [(None, None), (None, None)]

    def _eval_subroutine(self, z):
        pass

    def eval(self, z):
        """Method that evaluates the forward model at the specified parameters `z`.
        This method is implemented via a book-keeping strategy under the hood to improve efficiency
        when called from a Metropolis-Hastings schema.

        Parameters
        ----------
        z : float or numpy.ndarray
            Value of the parameters at which evaluation is required.

        Returns
        -------
        numpy.ndarray
            The evaluations of the forward model at every spatial location.

        Notes
        -----
        Because of book-keeping, not every call to this method will trigger active computation of the result
        (if the parameters have recently been queried, the recorded results will be returned at no additional cost).
        """
        if self.last_evaluations[0][0] is not None and np.all(
                np.abs(z - self.last_evaluations[0][0]) < np.finfo(np.float32).eps):
            pass
        # condition to avoid repeated evaluations for the same parameter
        elif self.last_evaluations[1][0] is not None and np.all(
                np.abs(z - self.last_evaluations[1][0]) < np.finfo(np.float32).eps):
            self.last_evaluations.reverse()
        else:
            self.last_evaluations[1] = (z, self._eval_subroutine(z))
            self.last_evaluations.reverse()
        return self.last_evaluations[0][1]

    def logposterior(self, z):
        """Computes the log posterior (up to an additive constant) at the provided set of forward model parameters.

        Parameters
        ----------
        z : float or numpy.ndarray
            Value of the parameters at which evaluation is required.

        Returns
        -------
        float
            The log-posterior (up to a constant).
        """
        predicted = self.eval(z)
        res = self.log_error_density(self.data - predicted) + self.log_prior(z)
        return res


