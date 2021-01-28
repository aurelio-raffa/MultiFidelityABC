import chaospy as cpy
import numpy as np
import progressbar as pb

from chaospy import Normal, generate_expansion

from source.models.surrogate import SurrogateModel
from source.utils.decorators import time_it


class PCESurrogate(SurrogateModel):
    """This class implements a low-fidelity surrogate, approximating the forward
    model through Polynomial Chaos Expansion.
    """
    def __init__(self, data, log_error_density, prior, log_prior, degree, multi_fidelity_q):
        """Parameters
        ----------
        data : numpy.ndarray
            Sample of observed values of the forward model in the spatial nodes.
        log_error_density : function or callable object
            It computes the log density at the input vector (used for computing the likelihood, passing
            as input the difference between the data and the vector of the evaluations of the surrogate model
            in the corresponding spatial nodes for a fixed  parameter `z`).
        prior : object
            Object that represents the prior of the parameter from chaospy.distributions.
        log_prior : function or callable object
            It computes the log prior at the input vector (of parameters).
        degree : int
            Degree of the polynomial expansions.
        multi_fidelity_q : int
            Number of points to be drawn when performing a multi-fidelity update.
        """
        super().__init__(data, log_error_density, prior, log_prior, multi_fidelity_q)
        self.degree = degree
        self.expansion = None
        self.proxy = None

    @staticmethod
    def _get_coeffs(fitted_poly):
        return np.concatenate([np.array(p.coefficients).reshape(1, -1) for p in fitted_poly], axis=0)

    @time_it(only_time=True)
    def fit(self, high_fidelity, num_evals=None, quadrature_rule='gaussian'):
        """Fits the low-fidelity surrogate via Polynomial Chaos Expansion.

        Parameters
        ----------
        high_fidelity : HighFidelityModel
            Model that we want to approximate through the low-fidelity.
        num_evals : int, default None
            Parameter provided for consistency, the actual number of evaluations is determined by the
            quadrature rule.
        quadrature_rule: str, default 'gaussian'
            Rule used for the quadrature (passed to chaospy.generate_quadrature.
        """
        abscissae, weights = cpy.generate_quadrature(self.degree, self.prior, rule=quadrature_rule)
        self.expansion = generate_expansion(self.degree, self.prior, retall=False)
        widgets = ['fit\t', pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=abscissae.T.shape[0], widgets=widgets)
        evals = []
        bar.start()
        for i, z_ in enumerate(abscissae.T):
            evals.append(high_fidelity.eval(z_))
            bar.update(i + 1)
        self.proxy = cpy.fit_quadrature(self.expansion, abscissae, weights, evals)
        self._fit = True

    def _eval_subroutine(self, z):
        return self.proxy(*z) if type(z) is np.ndarray else self.proxy(z)

    def multi_fidelity_update(self, y, radius, high_fidelity):
        new_points = super().multi_fidelity_update(y, radius, high_fidelity)
        new_evals = [high_fidelity.eval(z_) - self.eval(z_) for z_ in new_points.T]
        new_poly = cpy.fit_regression(self.expansion, new_points, new_evals)
        self.proxy = self.proxy + new_poly
