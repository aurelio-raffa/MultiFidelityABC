import numpy as np

from source.models.base_model import ForwardModel


class SurrogateModel(ForwardModel):
    """Base class implementing a surrogate model.

    This class defines the paradigm of use for every surrogate class in this framework.
    """
    def __init__(self, data, log_error_density, prior, log_prior, multi_fidelity_q):
        """Parameters
        ----------
        data : numpy.ndarray
            The sample of observed values of the forward model in the spatial nodes.
        log_error_density : function or callable object
            A function that returns the joint log density of the errors on a vector of values.
        prior : object
            Object that represents the prior of the parameter from chaospy.distributions.
        log_prior : function or callable object
            It computes the log prior at the input vector (of parameters).
        multi_fidelity_q : int
            Number of points to be drawn when performing a multi-fidelity update.
        """
        super().__init__(data, log_error_density, log_prior)
        self.prior = prior
        self.multi_fidelity_q = multi_fidelity_q
        self._fit = False

    def eval(self, z):
        assert self._fit
        return super().eval(z)

    def multi_fidelity_update(self, y, radius, high_fidelity):
        """Member function that updates the low-fidelity, based on the polynomial chaos
        expansion of the difference between the high and low fidelity models.
        It uses a regression method on new points drawn from a uniform
        distribution centered on y with radius passed as input.

        Parameters
        ----------
        y : float or numpy.ndarray
            Center used for drawing the sample.
        radius : float
            Value of the radius used for drawing the sample.
        high_fidelity : HighFidelityModel
            Model that we want to approximate with the low-fidelity.
        """
        if type(y) is np.ndarray:
            new_points = y.reshape(-1, 1) + np.random.uniform(-radius, radius, (y.shape[0], self.multi_fidelity_q))
        else:
            new_points = y + np.random.uniform(-radius, radius, self.multi_fidelity_q)
            new_points = new_points.reshape(1, -1)
        return new_points
